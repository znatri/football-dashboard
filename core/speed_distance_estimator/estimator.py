from typing import Optional, Dict, List
import sys

sys.path.append("../../")

from utils import measure_distance, get_foot_position

import cv2
import numpy as np


class PlayerPerformanceMetrics:
    def __init__(self, analysis_window: int = 5, frame_rate: int = 24):
        self.analysis_window = analysis_window
        self.frame_rate = frame_rate

    def calculate_and_append_metrics(self, tracks):
        """
        Calculates speed and distance metrics for each player.
        :param tracks: Dictionary of tracks for each object.
        """
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue

            number_of_frames = len(object_tracks)

            for frame_num in range(0, number_of_frames, self.analysis_window):
                last_frame = min(frame_num + self.analysis_window, number_of_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id][
                        "position_transformed"
                    ]
                    end_position = object_tracks[last_frame][track_id][
                        "position_transformed"
                    ]

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_mps = distance_covered / time_elapsed
                    speed_kmph = speed_mps * 3.6

                    if object not in total_distance:
                        total_distance[object] = {}

                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]["speed_kmph"] = (
                            speed_kmph
                        )
                        tracks[object][frame_num_batch][track_id]["distance_m"] = (
                            total_distance[object][track_id]
                        )

    def annotate_frames_with_metrics(self, frames, tracks):
        """
        Annotates frames with speed and distance metrics for each player.
        :param frames: List of frames to annotate.
        :param tracks: Dictionary of player tracks.
        :return: List of annotated frames.
        """
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue

                for track_id, track_info in object_tracks[frame_num].items():
                    speed = track_info.get("speed_kmph")
                    distance = track_info.get("distance_m")
                    if speed is None or distance is None:
                        continue

                    bbox = track_info["bbox"]
                    position = list(get_foot_position(bbox))
                    position[1] += (
                        40  # Offset for displaying text below the foot position
                    )

                    cv2.putText(
                        frame,
                        f"{speed:.2f} km/h",
                        (position[0], position[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"{distance:.2f} m",
                        (position[0], position[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                    )

            output_frames.append(frame)
        return output_frames
