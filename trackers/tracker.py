import os
import pickle
from typing import List, Dict, Union
import sys
sys.path.append("../ ")

from utils import get_center_of_bbox, get_bbox_width, get_foot_position

import cv2
import numpy as np
import supervision as sv
import pandas as pd
from ultralytics import YOLO


class ObjectTracker:
    '''
    Object tracker
    '''
    def __init__(self, model_path: str, tracker_type: str = 'ByteTrack'):
        """
        Initialize the tracker

        :param model_path: path to the model
        :param tracker_type: type of tracker to use (default is ByteTrack)
        """
        self.model = YOLO(model_path)
        self.tracker = self._initialize_tracker(tracker_type)

    def _initialize_tracker(self, tracker_type: str):
        """
        Initialize the tracker based on the tracker type.

        :param tracker_type: type of tracker to use
        :return: initialized tracker instance
        """
        if tracker_type == 'ByteTrack':
            return sv.ByteTrack()
        else:
            raise ValueError(f"Unsupported tracker type: {tracker_type}")
        
    def add_position_to_tracks(self, tracks):
        """
        Adds positional information to each track in the tracking dictionary. For the ball, the position is defined
        as the center of its bounding box. For players or other objects, the position is defined as the foot position
        derived from the bounding box.

        :param tracks: A dictionary where each key represents an object type (e.g., 'ball', 'players') and the value
                    is a list of dictionaries for each frame that map a track ID to track information including
                    the bounding box.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    try:
                        bbox = track_info['bbox']

                        # Determine position based on the type of object
                        if object == 'ball':
                            position= get_center_of_bbox(bbox)
                        else:
                            position = get_foot_position(bbox)
                            
                        tracks[object][frame_num][track_id]['position'] = position
                    except (KeyError, TypeError, IndexError) as error:
                        print(f"Error adjusting position for track ID {track_id} in frame {frame_num}: {error}")
                        continue
                    
    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_objects(self, frames: List) -> List:
        """
        Detect objects in a batch of frames

        :param frames: list of frames
        :return: list of detections
        """
        batch_size = 32
        detections = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            results = self.model(batch, conf=0.1)
            detections.extend(results)
        
        return detections

    def track_objects(self, frames: List, read_from_stub: bool = False, stub_path: Union[str, None] = None) -> Dict:
        """
        Track objects in a batch of frames

        :param frames: list of frames
        :param read_from_stub: whether to read tracks from a stub file
        :param stub_path: path to the stub file
        :return: dictionary containing the object tracks
        """

        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
            
        detections = self.detect_objects(frames)
        tracks = {
            "players": [{} for _ in range(len(detections))],
            "referees": [{} for _ in range(len(detections))],
            "ball": [{} for _ in range(len(detections))]
        }

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inverted = {v: k for k, v in class_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            for obj_indx, cls_id in enumerate(detection_supervision.class_id):
                if class_names[cls_id] == 'goalkeeper':
                    detection_supervision.class_id[obj_indx] = class_names_inverted['player']

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == class_names_inverted['player']:
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}

                elif cls_id == class_names_inverted['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}

                elif cls_id == class_names_inverted['ball']:
                    tracks['ball'][frame_num][1] = {'bbox': bbox}

        if stub_path:
            try:
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks, f)
            except IOError as e:
                raise RuntimeError(f"Error saving tracks to file: {e}")

        return tracks
    
    def draw_ellipse(self, frame, bbox: List, color, track_id=None):
        """
        Draw an ellipse on a frame

        :param frame: frame to draw on
        :param bbox: bounding box coordinates
        :param color: color of the ellipse
        :param track_id: ID of the object track
        :return: annotated frame
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame, 
            center=(x_center, y2), 
            axes=(int(width), int(0.35 * width)), 
            angle=0, 
            startAngle=-45, 
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        # Draw rectangle and track ID
        rectangle_width = int(40)
        rectangle_height = int(20)
        x1_rect =  int(x_center - rectangle_width // 2)
        y1_rect = int(y2 - rectangle_height // 2) + 15
        x2_rect = int(x_center + rectangle_width // 2)
        y2_rect = int(y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame, 
                (x1_rect, y1_rect), 
                (x2_rect, y2_rect),
                color,
                cv2.FILLED
            ) 

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                str(track_id),
                (x1_text, y1_rect + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            ) 

        return frame
    
    def draw_triangle(self, frame, bbox: List, color):
        """
        Draw a triangle on a frame

        :param frame: frame to draw on
        :param bbox: bounding box coordinates
        :param color: color of the triangle
        :return: annotated frame
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        
        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])

        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
    
    def draw_ball_control_stats(self, frame, frame_num, team_ball_control):
        """
        Draw ball control statistics on a video frame.

        :param frame: The current video frame to draw on.
        :param frame_num: The index of the current frame in the sequence.
        :param team_ball_control: An array indicating which team controlled the ball at each frame.
        :return: The frame with ball control statistics drawn on it.
        """
        # Define the area for displaying stats
        start_point = (1350, 850)
        end_point = (1900, 970)
        color_white = (255, 255, 255)
        transparency = 0.4
        
        # Create a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, start_point, end_point, color_white, cv2.FILLED)
        cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0, frame)

        # Calculate ball control percentages up to the current frame
        control_data_until_now = team_ball_control[:frame_num + 1]
        frames_with_team_1_control = np.sum(control_data_until_now == 1)
        frames_with_team_2_control = np.sum(control_data_until_now == 2)
        total_controlled_frames = frames_with_team_1_control + frames_with_team_2_control

        if total_controlled_frames > 0:
            team_1_percentage = frames_with_team_1_control / total_controlled_frames * 100
            team_2_percentage = frames_with_team_2_control / total_controlled_frames * 100
        else:
            team_1_percentage = team_2_percentage = 0

        # Draw the text on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Team 1 Ball Control: {team_1_percentage:.2f}%", (1400, 900), font, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2_percentage:.2f}%", (1400, 950), font, 1, (0, 0, 0), 3)

        return frame
    
    def draw_annotations(self, frames: List, tracks: Dict, team_ball_control: List):
        """
        Draw annotations on a batch of frames

        :param frames: list of frames
        :param tracks: dictionary containing the object tracks
        :param team_ball_control: An array indicating which team controlled the ball at each frame.
        :return: list of annotated frames
        """
        annotated_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            
            # Draw Player BBoxes
            for track_id, player in player_dict.items():
                color = player.get('team_color',  (0, 0, 255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0, 0, 255))

            # Draw Referee BBoxes
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255)) 

            # Draw Ball BBox
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"],(0,255,0))

            # Draw Team Ball Control
            frame = self.draw_ball_control_stats(frame, frame_num, team_ball_control)

            annotated_frames.append(frame)

        return annotated_frames