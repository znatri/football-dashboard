import os
import pickle
from typing import List
import sys

sys.path.append("../../")

from utils import measure_distance, measure_xy_distance

import cv2
import numpy as np


class CamerMovementEstimator:
    def __init__(self, frame) -> None:
        """
        Initialize the Camera Movement Estimator with the first frame.

        :param frame: The first video frame in BGR format.
        """
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 255  # Top 20 rows
        mask_features[:, 900:1050] = 255  # Last 150 rows from the bottom

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self.min_dist = 5

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        """
        Calculate the camera movement based on a list of frames.

        :param frames: List of frames to analyze.
        :param read_from_stub: If True, read precalculated movement from a file.
        :param stub_path: Path to the stub file if read_from_stub is True.
        :return: A list of tuples representing the x and y movements of the camera.
        """
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        camera_movement = [[0, 0] for _ in range(len(frames))]
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num, frame in enumerate(frames[1:], start=1):
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            if new_features is not None and old_features is not None:
                max_dist, camera_movement_x, camera_movement_y = 0, 0, 0

                for i, (new, old) in enumerate(zip(new_features, old_features)):
                    new, old = new.ravel(), old.ravel()
                    dist = measure_distance(new, old)
                    if dist > max_dist:
                        max_dist = dist
                        camera_movement_x, camera_movement_y = measure_xy_distance(
                            old, new
                        )

                if max_dist > self.min_dist:
                    camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                    old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        if stub_path:
            try:
                with open(stub_path, "wb") as f:
                    pickle.dump(camera_movement, f)
            except IOError as e:
                raise RuntimeError(f"Error saving tracks to file: {e}")

        return camera_movement

    def draw_camera_movement(
        self, frames: List[np.ndarray], camera_movement_per_frame: List
    ) -> List[np.ndarray]:
        """
        Annotate each frame with the camera's movement data.

        :param frames: List of frames to draw on.
        :param camera_movement_per_frame: List of tuples representing the x and y movements of the camera.
        :return: List of frames with the camera movement annotations.
        """
        output_frames = []

        for frame_num, frame in enumerate(frames):
            annotated_frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(
                overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame
            )

            x_movement, y_movement = camera_movement_per_frame[frame_num]

            cv2.putText(
                annotated_frame,
                f"Camera Movement X: {x_movement:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                annotated_frame,
                f"Camera Movement Y: {y_movement:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )

            output_frames.append(annotated_frame)

        return output_frames

    def adjust_position_to_tracks(self, tracks, camera_movement_per_frame) -> None:
        """
        Adjusts the positions of tracked objects based on the camera movement for each frame to provide a more stable
        tracking output that accounts for camera shifts.

        :param tracks: A dictionary with object types as keys. Each key points to a list of dictionaries (one per frame),
                    where each dictionary maps track IDs to track information including positions.
        :param camera_movement_per_frame: A list of tuples, where each tuple contains the x and y camera movement values for a frame.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    try:
                        position = track_info["position"]
                        camera_movement = camera_movement_per_frame[frame_num]
                        position_adjusted = (
                            position[0] - camera_movement[0],
                            position[1] - camera_movement[1],
                        )
                        tracks[object][frame_num][track_id]["position_adjusted"] = (
                            position_adjusted
                        )
                    except (KeyError, TypeError, IndexError) as error:
                        print(
                            f"Error adjusting position for track ID {track_id} in frame {frame_num}: {error}"
                        )
                        continue
