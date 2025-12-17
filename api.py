import json
import os
import uuid
from typing import Dict, Optional, List

import cv2
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core import ObjectTracker
from core.match_processor import MatchProcessingConfig, process_match

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_ROOT, exist_ok=True)


class ProcessRequest(BaseModel):
    mode: str = "pickup_mvp"
    fps: int = 15
    annotated_video: bool = True


app = FastAPI(title="Football Analytics Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobStatus(BaseModel):
    status: str
    progress: float
    message: str
    match_id: str


jobs: Dict[str, JobStatus] = {}

DEFAULT_MODEL_NAME = "best_train9.pt"


def _match_dir(match_id: str) -> str:
    return os.path.join(DATA_ROOT, match_id)


def _ensure_match_exists(match_id: str) -> str:
    mdir = _match_dir(match_id)
    if not os.path.isdir(mdir):
        raise HTTPException(status_code=404, detail="match_id not found")
    return mdir


@app.post("/api/matches")
async def create_match(file: UploadFile = File(...)):
    """Create a new match, save original video, and generate preview frame."""
    match_id = str(uuid.uuid4())
    mdir = _match_dir(match_id)
    os.makedirs(mdir, exist_ok=True)

    video_path = os.path.join(mdir, "original.mp4")

    # Save uploaded video stream to disk
    with open(video_path, "wb") as out_f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)

    # Generate preview.jpg from first frame
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise HTTPException(status_code=400, detail="Could not read video for preview")

    preview_path = os.path.join(mdir, "preview.jpg")
    cv2.imwrite(preview_path, frame)

    return {
        "match_id": match_id,
        "preview_url": f"/api/matches/{match_id}/preview",
    }


@app.get("/api/matches/{match_id}/preview")
def get_preview(match_id: str):
    """Return preview frame image bytes."""
    mdir = _ensure_match_exists(match_id)
    preview_path = os.path.join(mdir, "preview.jpg")
    if not os.path.exists(preview_path):
        raise HTTPException(status_code=404, detail="preview not found")
    return FileResponse(preview_path, media_type="image/jpeg")


@app.post("/api/matches/{match_id}/calibration")
async def submit_calibration(match_id: str, body: dict):
    """
    Store calibration JSON as provided by the frontend.
    Schema:
    {
      "pixel_points": [[x,y] * 4],
      "field_points_m": [[x,y] * 4],
      "field_size": { "width_m": W, "height_m": H },
      "notes": { ... }
    }
    """
    mdir = _ensure_match_exists(match_id)
    calib_path = os.path.join(mdir, "calib.json")
    with open(calib_path, "w") as f:
        json.dump(body, f)
    return {"status": "ok"}


@app.post("/api/matches/{match_id}/teams")
async def submit_teams(match_id: str, body: dict):
    """
    Store team labels JSON as provided by the frontend.
    Schema:
    { "labels": { "track_id_12": "A", ... } }
    """
    mdir = _ensure_match_exists(match_id)
    team_path = os.path.join(mdir, "team_labels.json")
    with open(team_path, "w") as f:
        json.dump(body, f)
    return {"status": "ok"}


@app.get("/api/matches/{match_id}/team-samples")
def get_team_samples(match_id: str, max_samples: int = 8, max_frames: int = 30):
    """
    Return a small set of player thumbnail images from early frames to help UI labeling.

    Implementation:
    - run detection + tracking on first N frames
    - extract top-half crops of player bboxes
    - store as data/{match_id}/crops/{track_id}.jpg
    - return list of {track_id, crop_url}
    """
    mdir = _ensure_match_exists(match_id)
    video_path = os.path.join(mdir, "original.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=400, detail="original.mp4 not found for match")

    crops_dir = os.path.join(mdir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    # Initialize tracker once
    model_path = os.path.join("models", DEFAULT_MODEL_NAME)
    tracker = ObjectTracker(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video for samples")

    samples: Dict[int, str] = {}
    frame_idx = 0

    try:
        while len(samples) < max_samples and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_idx += 1

            results = tracker.model([frame], conf=0.2)
            if not results:
                continue
            result = results[0]

            class_names = result.names
            from supervision import Detections

            det_sv = Detections.from_ultralytics(result)

            # Track detections to obtain stable track IDs
            tracked_dets = tracker.tracker.update_with_detections(det_sv)

            for det in tracked_dets:
                bbox = det[0].tolist()
                cls_id = det[3]
                track_id = int(det[4])

                if class_names[cls_id] != "player":
                    continue

                if track_id in samples:
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                y_mid = y1 + (y2 - y1) // 2
                y2_crop = max(y_mid, y1 + 1)
                crop = frame[y1:y2_crop, x1:x2]
                if crop.size == 0:
                    continue

                crop_path = os.path.join(crops_dir, f"{track_id}.jpg")
                cv2.imwrite(crop_path, crop)
                samples[track_id] = f"/api/matches/{match_id}/crops/{track_id}"

                if len(samples) >= max_samples:
                    break

    finally:
        cap.release()

    samples_list = [
        {"track_id": int(tid), "crop_url": url} for tid, url in samples.items()
    ]
    return {"samples": samples_list}


def _run_processing_job(job_id: str, match_id: str, req: ProcessRequest):
    def update_progress(p: float) -> None:
        status = jobs.get(job_id)
        if status:
            status.progress = float(max(0.0, min(1.0, p)))
            jobs[job_id] = status

    try:
        jobs[job_id] = JobStatus(
            status="running",
            progress=0.0,
            message="Processing started",
            match_id=match_id,
        )

        cfg = MatchProcessingConfig(
            fps_processed=req.fps,
            annotated_video=req.annotated_video,
            mode=req.mode,
        )
        process_match(match_id=match_id, config=cfg, data_root=DATA_ROOT, progress_cb=update_progress)

        status = jobs[job_id]
        status.status = "succeeded"
        status.progress = 1.0
        status.message = "Processing finished"
        jobs[job_id] = status
    except Exception as e:
        status = jobs.get(job_id)
        message = str(e)
        if status:
            status.status = "failed"
            status.message = message
            status.progress = status.progress or 0.0
            jobs[job_id] = status
        else:
            jobs[job_id] = JobStatus(
                status="failed",
                progress=0.0,
                message=message,
                match_id=match_id,
            )


@app.post("/api/matches/{match_id}/process")
async def start_processing(
    match_id: str, req: ProcessRequest, background_tasks: BackgroundTasks
):
    """Start asynchronous processing job for a match."""
    _ensure_match_exists(match_id)
    video_path = os.path.join(_match_dir(match_id), "original.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=400, detail="original.mp4 not found for match")

    job_id = str(uuid.uuid4())
    jobs[job_id] = JobStatus(
        status="queued",
        progress=0.0,
        message="Queued",
        match_id=match_id,
    )

    background_tasks.add_task(_run_processing_job, job_id, match_id, req)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    status = jobs.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="job_id not found")
    return status


@app.get("/api/matches/{match_id}/results/summary")
def get_summary(match_id: str):
    mdir = _ensure_match_exists(match_id)
    summary_path = os.path.join(mdir, "results", "summary.json")
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="summary not found")
    with open(summary_path, "r") as f:
        return json.load(f)


@app.get("/api/matches/{match_id}/results/highlights")
def get_highlights(match_id: str):
    mdir = _ensure_match_exists(match_id)
    highlights_path = os.path.join(mdir, "results", "highlights.json")
    if not os.path.exists(highlights_path):
        raise HTTPException(status_code=404, detail="highlights not found")
    with open(highlights_path, "r") as f:
        return json.load(f)


@app.get("/api/matches/{match_id}/results/annotated")
def get_annotated_video(match_id: str):
    mdir = _ensure_match_exists(match_id)
    annotated_path = os.path.join(mdir, "results", "annotated.mp4")
    if not os.path.exists(annotated_path):
        raise HTTPException(status_code=404, detail="annotated video not found")
    return FileResponse(annotated_path, media_type="video/mp4")


@app.get("/api/matches/{match_id}/crops/{crop_id}")
def get_crop(match_id: str, crop_id: str):
    """Serve stored player crop images."""
    mdir = _ensure_match_exists(match_id)
    crop_path = os.path.join(mdir, "crops", f"{crop_id}.jpg")
    if not os.path.exists(crop_path):
        raise HTTPException(status_code=404, detail="crop not found")
    return FileResponse(crop_path, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

