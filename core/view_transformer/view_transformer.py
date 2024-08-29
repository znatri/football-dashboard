from typing import Optional, Dict, List

import cv2
import numpy as np


class ViewTransformer:
    def __init__(
        self,
    ):
        court_width = 68
        court_length = 23.32

        self.pixel_vertices = np.array(
            [[110, 1035], [265, 275], [910, 260], [1640, 915]], dtype=np.float32
        )

        self.target_vertices = np.array(
            [[0, court_width], [0, 0], [court_length, 0], [court_length, court_width]],
            dtype=np.float32,
        )

        # Ensure arrays are correctly formatted before transformation
        if self.pixel_vertices.shape[0] != self.target_vertices.shape[0]:
            raise ValueError(
                "Number of pixel vertices must match number of target vertices."
            )

        self.perspective_transform = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point: List[float]) -> Optional[np.ndarray]:
        """
        Transforms a point from image coordinates to court coordinates using the calculated perspective transform.

        :param point: A list with x and y coordinates of the point.
        :return: Transformed coordinates as a numpy ndarray or None if the point is outside the defined vertices.
        """
        p = np.array([point], dtype=np.float32)
        is_inside = (
            cv2.pointPolygonTest(self.pixel_vertices, (p[0][0], p[0][1]), False) >= 0
        )
        if not is_inside:
            return None

        transformed_point = cv2.perspectiveTransform(
            p.reshape(-1, 1, 2), self.perspective_transform
        )
        return transformed_point.reshape(-1, 2)

    def add_transformed_positions_to_tracks(self, tracks: Dict):
        """
        Adds transformed positions to each track in the tracks dictionary.

        :param tracks: A dictionary with track data structured by object type and frames.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    try:
                        position = track_info["position_adjusted"]
                        position_transformed = self.transform_point(position)
                        if position_transformed is not None:
                            position_transformed = (
                                position_transformed.squeeze().tolist()
                            )
                        tracks[object][frame_num][track_id]["position_transformed"] = (
                            position_transformed
                        )
                    except (KeyError, TypeError, IndexError) as error:
                        print(
                            f"Error adjusting position for track ID {track_id} in frame {frame_num}: {error}"
                        )
                        continue
