import os
import sys
import traceback
import time
import argparse

import numpy as np
from core import (
    ObjectTracker,
    TeamAssigner,
    PlayerBallAssigner,
    CamerMovementEstimator,
    ViewTransformer,
    PlayerPerformanceMetrics,
)
from utils import read_video, save_video, configure_logging, generate_output_filename
import torch


def setup_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def list_models(directory="models"):
    return [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]


def main(runtime_args, log):
    input_video = runtime_args.input_video or "input_videos/08fd33_4.mp4"
    output_video = runtime_args.output_path or generate_output_filename(input_video)

    if runtime_args.verbose:
        log.info(f"Start processing video: {input_video}")

    # Read Video
    video_frames = read_video(input_video)
    if runtime_args.verbose:
        log.debug("Video read: OK")

    # Get Tracks
    model_path = os.path.join("models", runtime_args.model)
    log.info(f"Using model: {model_path}")
    tracker = ObjectTracker(model_path)
    tracks = tracker.track_objects(
        video_frames,
        read_from_stub=runtime_args.use_stubs,
        stub_path="stubs/object_tracker_stubs.pkl",
    )
    if runtime_args.verbose:
        log.debug("Object tracking: OK")

    # Add Positions to Tracks
    tracker.add_position_to_tracks(tracks)
    if runtime_args.verbose:
        log.debug("Player positions added: OK")

    # Camera Movement Estimator
    camera_movement_estimator = CamerMovementEstimator(frame=video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=runtime_args.use_stubs,
        stub_path="stubs/camera_movement_stubs.pkl",
    )
    camera_movement_estimator.adjust_position_to_tracks(
        tracks, camera_movement_per_frame
    )
    if runtime_args.verbose:
        log.debug("Camera movement adjusted: OK")

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_positions_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    if runtime_args.verbose:
        log.debug("Ball positions interpolated: OK")

    # Speed and Distance Estimator
    speed_and_distance_estimator = PlayerPerformanceMetrics()
    speed_and_distance_estimator.calculate_and_append_metrics(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_colors(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )
    if runtime_args.verbose:
        log.debug("Player teams assigned: OK")

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assigner_ball_to_player(
            players=player_track, ball_bbox=ball_bbox
        )

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(
                tracks["players"][frame_num][assigned_player]["team"]
            )
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)
    if runtime_args.verbose:
        log.debug("Ball control assigned: OK")

    # Draw Output
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control
    )
    if runtime_args.verbose:
        log.debug("Annotations drawn: OK")
    del tracker

    # Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )
    if runtime_args.verbose:
        log.debug("Camera movement drawn: OK")
    del camera_movement_estimator

    # Draw Speed and Distance
    output_video_frames = speed_and_distance_estimator.annotate_frames_with_metrics(
        output_video_frames, tracks
    )
    if runtime_args.verbose:
        log.debug("Speed and distance metrics drawn: OK")
    del speed_and_distance_estimator

    # Save Video
    save_video(output_video_frames, output_video)
    if runtime_args.verbose:
        log.info(f"Video saved: {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process video with optional stub usage."
    )
    parser.add_argument(
        "-i", "--input_video", type=str, help="Path to the input video file"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, help="Path to save the output video file"
    )
    parser.add_argument(
        "--use_stubs",
        action="store_true",
        help="Use stubs for tracking and camera movement estimation",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbosity flag for logging"
    )
    parser.add_argument(
        "-s",
        "--snapshot",
        action="store_true",
        help="Capture snapshot of CUDA memory usage",
    )
    parser.add_argument(
        "-l", "--list_models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="best_train9.pt",
        help="Specify which model to use (default: best.pt)",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for model in list_models():
            print(model)
        sys.exit(0)

    try:
        setup_environment()
        log = configure_logging("ml-pipeline", args.verbose)

        if args.snapshot:
            import torch

            torch.cuda.memory._record_memory_history(stacks="all")  # Log CUDA memory
            torch.cuda.empty_cache()

        main(args, log)

    except Exception as e:
        exc_info = sys.exc_info()
        log.error(f'{e}\n{"%"*25} ERROR {"%"*25}\n')
        traceback.print_exception(*exc_info)
        log.error(f'{"%"*60}\n')

    finally:
        if args.snapshot:
            import torch

            logname = f"memdump/{time.time()}.pickle"
            torch.cuda.memory._dump_snapshot(logname)
            log.info(f"Saved memory log: {logname}")
