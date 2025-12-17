import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from core import ObjectTracker, PlayerBallAssigner, TeamAssigner
from core.calibration import Calibration
from utils import configure_logging, get_foot_position, get_center_of_bbox, measure_distance


@dataclass
class MatchProcessingConfig:
    """Configuration for processing a single match."""

    fps_processed: int = 15
    annotated_video: bool = True
    mode: str = "pickup_mvp"
    model: str = "best_train9.pt"


def _load_calibration(calib_path: str) -> Optional[Dict]:
    if not os.path.exists(calib_path):
        return None
    with open(calib_path, "r") as f:
        return json.load(f)


def _load_team_labels(team_labels_path: str) -> Dict[str, str]:
    if not os.path.exists(team_labels_path):
        return {}
    with open(team_labels_path, "r") as f:
        data = json.load(f)
    # Expected schema: { "labels": { "track_id_12": "A", ... } }
    return data.get("labels", {})


def _estimate_ball_quality(tracks: Dict) -> str:
    """Estimate ball tracking quality based on how many frames contain a ball point."""
    ball_tracks: List[Dict] = tracks.get("ball", [])
    if not ball_tracks:
        return "low"

    total_frames = len(ball_tracks)
    frames_with_ball = sum(1 for frame in ball_tracks if frame.get("has_ball_point"))
    ratio = frames_with_ball / max(total_frames, 1)

    if ratio >= 0.8:
        return "high"
    if ratio >= 0.5:
        return "medium"
    return "low"


def _build_possession_timeline(
    team_ball_control: np.ndarray, fps_processed: int
) -> List[Dict]:
    """Convert per-frame team control (1/2/0) into timeline segments."""
    if team_ball_control.size == 0:
        return []

    segments: List[Dict] = []
    current_team = int(team_ball_control[0])
    start_idx = 0

    for i in range(1, len(team_ball_control)):
        team = int(team_ball_control[i])
        if team != current_team:
            segments.append(
                {
                    "t0": start_idx / fps_processed,
                    "t1": i / fps_processed,
                    "team": "A" if current_team == 1 else "B" if current_team == 2 else "unknown",
                    "confidence": 0.7 if current_team in (1, 2) else 0.0,
                }
            )
            current_team = team
            start_idx = i

    # Last segment
    segments.append(
        {
            "t0": start_idx / fps_processed,
            "t1": len(team_ball_control) / fps_processed,
            "team": "A" if current_team == 1 else "B" if current_team == 2 else "unknown",
            "confidence": 0.7 if current_team in (1, 2) else 0.0,
        }
    )
    return segments


def _build_highlights_from_speeds(
    tracks: Dict, fps_processed: int, top_n: int = 5
) -> List[Dict]:
    """
    Very simple highlight heuristic:
    - Mark frames where any player's speed exceeds a threshold.
    - Cluster contiguous frames into segments and score by max speed in the segment.
    """
    speed_threshold_kmph = 20.0
    players_tracks = tracks.get("players", [])
    if not players_tracks:
        return []

    high_energy_frames: List[Tuple[int, float]] = []
    for frame_idx, frame_tracks in enumerate(players_tracks):
        max_speed = 0.0
        for _, info in frame_tracks.items():
            speed = float(info.get("speed_kmph", 0.0))
            if speed > max_speed:
                max_speed = speed
        if max_speed >= speed_threshold_kmph:
            high_energy_frames.append((frame_idx, max_speed))

    if not high_energy_frames:
        return []

    # Cluster contiguous frames
    segments: List[Tuple[int, int, float]] = []
    seg_start, seg_max_speed = high_energy_frames[0][0], high_energy_frames[0][1]
    prev_frame = high_energy_frames[0][0]

    for frame_idx, max_speed in high_energy_frames[1:]:
        if frame_idx == prev_frame + 1:
            seg_max_speed = max(seg_max_speed, max_speed)
        else:
            segments.append((seg_start, prev_frame, seg_max_speed))
            seg_start, seg_max_speed = frame_idx, max_speed
        prev_frame = frame_idx

    segments.append((seg_start, prev_frame, seg_max_speed))

    # Sort by score (max speed) and keep top N
    segments.sort(key=lambda x: x[2], reverse=True)
    segments = segments[:top_n]

    highlights: List[Dict] = []
    for start_f, end_f, score in segments:
        highlights.append(
            {
                "t0": start_f / fps_processed,
                "t1": (end_f + 1) / fps_processed,
                "score": float(score),
                "reason": "player_sprint",
            }
        )
    return highlights


def process_match(
    match_id: str,
    config: MatchProcessingConfig,
    data_root: str,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> None:
    """
    High-level processing entry point used by the FastAPI backend.

    This function:
    - streams video frames from disk (no full video in RAM)
    - runs detection + tracking and writes per-frame tracks.jsonl
    - aggregates player stats and possession timeline from these tracks
    - optionally writes annotated video
    - produces summary.json and highlights.json in the match results folder.
    """
    match_dir = os.path.join(data_root, match_id)
    video_path = os.path.join(match_dir, "original.mp4")
    calib_path = os.path.join(match_dir, "calib.json")
    team_labels_path = os.path.join(match_dir, "team_labels.json")
    results_dir = os.path.join(match_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    log = configure_logging(f"match-{match_id}", verbose=False)
    log.info("Starting match processing")

    calib = _load_calibration(calib_path)
    if calib is None:
        raise RuntimeError(
            "Calibration file calib.json is required for processing but was not found."
        )
    team_labels = _load_team_labels(team_labels_path)

    # Build calibration transform (pixel -> field meters)
    calibration = Calibration(
        pixel_points=calib["pixel_points"],
        field_points_m=calib["field_points_m"],
    )

    # Open video and prepare streaming processing
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for processing.")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or float(config.fps_processed)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps_processed = float(config.fps_processed or native_fps)
    frame_stride = max(int(round(native_fps / fps_processed)), 1)

    # Initialize tracker and team assigner
    model_path = os.path.join("models", config.model)
    tracker = ObjectTracker(model_path)
    team_assigner = TeamAssigner()
    teams_initialized = False

    # For annotated video (optional)
    annotated_video_url = None
    writer = None
    visualizer = None
    if config.annotated_video:
        from core.visualizer import MatchVisualizer

        visualizer = MatchVisualizer()

    tracks_jsonl_path = os.path.join(results_dir, "tracks.jsonl")
    tracks_file = open(tracks_jsonl_path, "w")

    # Aggregators for stats
    player_stats: Dict[int, Dict] = {}
    frames_with_ball: Dict[int, List[int]] = {}
    team_ball_control: List[int] = []
    ball_presence_flags: List[bool] = []

    last_player_pos: Dict[int, Tuple[float, float]] = {}
    last_player_t: Dict[int, float] = {}

    # Streaming loop
    frame_idx = -1
    processed_idx = -1
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_stride != 0:
                continue

            processed_idx += 1
            t = processed_idx / fps_processed

            if progress_cb and frame_count > 0 and processed_idx % max(
                frame_count // 20, 1
            ) == 0:
                progress_cb(0.05 + 0.5 * (processed_idx / max(frame_count, 1)))

            # Run detection on this frame
            results = tracker.model([frame], conf=0.1)
            if not results:
                continue
            result = results[0]

            class_names = result.names
            class_names_inverted = {v: k for k, v in class_names.items()}

            det_sv = sv.Detections.from_ultralytics(result)
            # Normalize goalkeeper -> player as in original pipeline
            for obj_idx, cls_id in enumerate(det_sv.class_id):
                if class_names[cls_id] == "goalkeeper":
                    det_sv.class_id[obj_idx] = class_names_inverted["player"]

            tracked_dets = tracker.tracker.update_with_detections(det_sv)

            # Per-frame structures in field coordinates
            players_frame_field: Dict[int, Tuple[float, float]] = {}
            ball_point_field: Optional[Tuple[float, float]] = None
            ball_confidence: Optional[float] = None

            # Extract players + ball and map to field coords
            for det in tracked_dets:
                bbox = det[0].tolist()
                cls_id = det[3]
                track_id = int(det[4])
                conf = float(det[2]) if len(det) > 2 else 1.0

                if class_names[cls_id] == "player":
                    foot_px = get_foot_position(bbox)
                    X, Y = calibration.pixel_to_field(foot_px[0], foot_px[1])
                    players_frame_field[track_id] = (X, Y)

                elif class_names[cls_id] == "ball":
                    center_px = get_center_of_bbox(bbox)
                    Xb, Yb = calibration.pixel_to_field(center_px[0], center_px[1])
                    ball_point_field = (Xb, Yb)
                    ball_confidence = conf

            # Initialize teams on first frame with players
            if players_frame_field and not teams_initialized:
                first_players_dict = {}
                for det in tracked_dets:
                    bbox = det[0].tolist()
                    cls_id = det[3]
                    track_id = int(det[4])
                    if class_names[cls_id] == "player":
                        first_players_dict[track_id] = {"bbox": bbox}
                if first_players_dict:
                    team_assigner.assign_team_colors(frame, first_players_dict)
                    teams_initialized = True

            # Determine team and update per-player stats
            for pid, (X, Y) in players_frame_field.items():
                pid_int = int(pid)
                if pid_int not in player_stats:
                    player_stats[pid_int] = {
                        "track_id": pid_int,
                        "distance_m": 0.0,
                        "top_speed_mps": 0.0,
                        "team_auto": 0,
                    }

                # Distance & speed in meters
                if pid_int in last_player_pos:
                    dt = t - last_player_t[pid_int]
                    if dt > 0:
                        dist = measure_distance(last_player_pos[pid_int], (X, Y))
                        player_stats[pid_int]["distance_m"] += dist
                        speed_mps = dist / dt
                        if speed_mps > player_stats[pid_int]["top_speed_mps"]:
                            player_stats[pid_int]["top_speed_mps"] = speed_mps

                last_player_pos[pid_int] = (X, Y)
                last_player_t[pid_int] = t

                # Auto team from colors if available
                if teams_initialized and player_stats[pid_int]["team_auto"] == 0:
                    for det in tracked_dets:
                        bbox = det[0].tolist()
                        cls_id = det[3]
                        track_det_id = int(det[4])
                        if track_det_id == pid_int and class_names[cls_id] == "player":
                            team_id = team_assigner.get_player_team(
                                frame, bbox, pid_int
                            )
                            player_stats[pid_int]["team_auto"] = team_id
                            break

            # Ball possession per frame using meters
            ball_assigner = PlayerBallAssigner()
            if ball_point_field is not None and players_frame_field:
                players_for_ball = {
                    pid: {"position_field": pos}
                    for pid, pos in players_frame_field.items()
                }
                assigned_player = ball_assigner.assigner_ball_to_player(
                    players=players_for_ball, ball_position_field=ball_point_field
                )
            else:
                assigned_player = -1

            if assigned_player != -1:
                team_id = int(player_stats[int(assigned_player)].get("team_auto", 0))
                team_ball_control.append(team_id if team_id in (1, 2) else 0)
                frames_with_ball.setdefault(int(assigned_player), []).append(
                    processed_idx
                )
            else:
                team_ball_control.append(0)

            ball_presence_flags.append(ball_point_field is not None)

            # Write tracks.jsonl line
            players_out = {
                str(pid): {"X": float(pos[0]), "Y": float(pos[1])}
                for pid, pos in players_frame_field.items()
            }
            ball_out = (
                {
                    "X": float(ball_point_field[0]),
                    "Y": float(ball_point_field[1]),
                    "confidence": float(ball_confidence or 0.0),
                }
                if ball_point_field is not None
                else None
            )

            record = {
                "frame_index": processed_idx,
                "timestamp": float(t),
                "players": players_out,
                "ball": ball_out,
            }
            tracks_file.write(json.dumps(record) + "\n")

            # Annotated video frame (optional)
            if visualizer is not None:
                players_for_vis = {}
                ball_for_vis = {}
                for det in tracked_dets:
                    bbox = det[0].tolist()
                    cls_id = det[3]
                    track_id = int(det[4])
                    if class_names[cls_id] == "player":
                        players_for_vis[track_id] = {"bbox": bbox}
                    elif class_names[cls_id] == "ball":
                        ball_for_vis = {"bbox": bbox}

                team_possession = team_ball_control[-1] if team_ball_control else 0

                ann_frame = visualizer.draw_annotations(
                    frame,
                    {"players": players_for_vis},
                    ball_for_vis,
                    team_possession,
                )

                if writer is None:
                    h, w = ann_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    output_video_path = os.path.join(results_dir, "annotated.mp4")
                    writer = cv2.VideoWriter(
                        output_video_path, fourcc, fps_processed, (w, h)
                    )
                    annotated_video_url = (
                        f"/api/matches/{match_id}/results/annotated"
                    )

                writer.write(ann_frame)

        cap.release()
        tracks_file.close()

        if writer is not None:
            writer.release()

    finally:
        if cap.isOpened():
            cap.release()
        try:
            tracks_file.close()
        except Exception:
            pass
        if writer is not None:
            writer.release()

    total_frames = processed_idx + 1 if processed_idx >= 0 else 0
    duration_sec = total_frames / fps_processed if fps_processed > 0 else 0.0

    # Aggregate per-player summary from streaming stats
    players_summary: List[Dict] = []
    for pid, stats in player_stats.items():
        # Team label resolution: override auto labels with explicit /teams mapping if provided.
        auto_team_id = int(stats.get("team_auto", 0))
        default_team_label = (
            "A" if auto_team_id == 1 else "B" if auto_team_id == 2 else "unknown"
        )
        override_label = team_labels.get(f"track_id_{pid}")
        team_label = override_label or default_team_label

        distance_m = float(stats.get("distance_m", 0.0))
        top_speed_mps = float(stats.get("top_speed_mps", 0.0))
        avg_speed_mps = distance_m / duration_sec if duration_sec > 0 else 0.0

        # Possession seconds and touches estimate
        ball_frames = frames_with_ball.get(pid, [])
        possession_sec = len(ball_frames) / fps_processed if fps_processed > 0 else 0.0

        touches_est = 0
        last_idx = -10**9
        for idx in sorted(ball_frames):
            if idx - last_idx > fps_processed * 0.5:
                touches_est += 1
            last_idx = idx

        players_summary.append(
            {
                "track_id": pid,
                "team": team_label,
                "distance_m": float(distance_m),
                "top_speed_mps": float(top_speed_mps),
                "avg_speed_mps": float(avg_speed_mps),
                "heatmap_url": f"/api/matches/{match_id}/results/heatmaps/{pid}",
                "touches_est": int(touches_est),
                "possession_sec": float(possession_sec),
            }
        )

    # Timeline & highlights
    team_ball_control_arr = np.array(team_ball_control, dtype=int)
    possession_segments = _build_possession_timeline(
        team_ball_control_arr, fps_processed=int(fps_processed)
    )
    ball_tracks = [{"has_ball_point": f} for f in ball_presence_flags]
    ball_quality = _estimate_ball_quality({"ball": ball_tracks})
    highlights = []  # highlight heuristic can be added later using tracks.jsonl

    # Meta
    field_size_m = {"width_m": 60, "height_m": 40}
    if calib and "field_size" in calib:
        field_size_m = {
            "width_m": float(calib["field_size"].get("width_m", field_size_m["width_m"])),
            "height_m": float(
                calib["field_size"].get("height_m", field_size_m["height_m"])
            ),
        }

    summary = {
        "match_id": match_id,
        "meta": {
            "fps_processed": config.fps_processed,
            "duration_sec": float(duration_sec),
            "field_size_m": field_size_m,
            "camera": calib.get("notes", {}).get("camera", "fence_gopro_behind_goal")
            if calib
            else "fence_gopro_behind_goal",
            "ball_quality": ball_quality,
        },
        "players": players_summary,
        "timeline": {"possession_segments": possession_segments},
        "assets": {"annotated_video_url": annotated_video_url},
    }

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f)

    with open(os.path.join(results_dir, "highlights.json"), "w") as f:
        json.dump({"highlights": highlights}, f)

    if progress_cb:
        progress_cb(1.0)

