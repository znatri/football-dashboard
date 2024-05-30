import sys, traceback, time, logging # for debug
import argparse 
import numpy as np
from core import ObjectTracker, TeamAssigner, PlayerBallAssigner, CamerMovementEstimator, ViewTransformer, PlayerPerformanceMetrics
from utils import read_video, save_video
import torch

def main(runtime_args, log):
    torch.cuda.memory._record_memory_history()  # log cuda memory

    # Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')
    if runtime_args.verbose: log.info("video read: OK")

    # Get Tracks

    tracker = ObjectTracker('models/best.pt')
    tracks = tracker.track_objects(video_frames, read_from_stub=runtime_args.use_stubs,
                                stub_path='stubs/object_tracker_stubs.pkl')
    if runtime_args.verbose: log.info("get tracks: OK")

    # Get object positions
    tracker.add_position_to_tracks(tracks)
    if runtime_args.verbose: log.info("add player positon: OK")

    # Camera Movement Estimator
    camera_movement_estimator = CamerMovementEstimator(frame=video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=runtime_args.use_stubs,
                                                                              stub_path='stubs/camera_movement_stubs.pkl')
    camera_movement_estimator.adjust_position_to_tracks(tracks, camera_movement_per_frame)
    if runtime_args.verbose: log.info("adjust player positon: OK")

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_positions_to_tracks(tracks)

    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    if runtime_args.verbose: log.info("interpolate ball positon: OK")

    # Speed and distance estimator
    speed_and_distance_estimator = PlayerPerformanceMetrics()
    speed_and_distance_estimator.calculate_and_append_metrics(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_colors(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    if runtime_args.verbose: log.info("assign player team: OK")

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assigner_ball_to_player(players=player_track, ball_bbox=ball_bbox)

        # Team Ball Control is not accurate
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)
    if runtime_args.verbose: log.info("team ball control: OK")

    # Draw Output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    if runtime_args.verbose: log.info("assign player team: OK")

    # Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    if runtime_args.verbose: log.info("draw camera movement: OK")

    # Draw Speed and Distance
    speed_and_distance_estimator.annotate_frames_with_metrics(output_video_frames, tracks)
    if runtime_args.verbose: log.info("speed and distance estimator: OK")

    # Save Video
    save_video(output_video_frames, 'output_videos/08fd33_4.mp4')
    if runtime_args.verbose: log.info("save video: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video with optional stub usage.")
    parser.add_argument('--use_stubs', action='store_true',
                        help='Use stubs for tracking and camera movement estimation')
    parser.add_argument('-v','--verbose', action='store_true',
                        help='Verbosity flag for logging')
    args = parser.parse_args()
    
    try: # run pipeline
        logging.basicConfig(level=logging.INFO)
        log=logging.getLogger("ml-pipeline")
        main(args, log)

    except Exception as e: # catch failure
        exc_info = sys.exc_info()
        log.error(f'\n{e}\n%%%%%%%%%%%%%%%%%%%ERROR%%%%%%%%%%%%%%%%%%%\n')
        traceback.print_exception(*exc_info)
        log.error(f'\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    
    finally: # log memory
        logname = f"memdump/{time.time()}.pickle"
        torch.cuda.memory._dump_snapshot(logname)
        log.info(f"saved mem log: {logname}")
    