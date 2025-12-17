from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


class Calibration:
    """
    Encapsulates pixel↔field homography for a single calibrated camera.

    Construct with 4 corresponding pixel and field points (in meters), then
    use `pixel_to_field` / `transform_many` to map detections into field space.
    """

    def __init__(
        self,
        pixel_points: Sequence[Sequence[float]],
        field_points_m: Sequence[Sequence[float]],
    ):
        if len(pixel_points) != 4 or len(field_points_m) != 4:
            raise ValueError("Calibration requires exactly 4 pixel and 4 field points")

        src = np.asarray(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
        dst = np.asarray(field_points_m, dtype=np.float32).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src, dst, method=0)
        if H is None:
            raise ValueError("cv2.findHomography failed for provided calibration points")

        self.H = H

    def pixel_to_field(self, x: float, y: float) -> Tuple[float, float]:
        """Transform a single pixel point (x, y) into field coordinates (X, Y) in meters."""
        pts = np.array([[[x, y]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts, self.H)
        X, Y = warped[0, 0]
        return float(X), float(Y)

    def transform_many(
        self, points: Iterable[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Transform many pixel points into field coordinates.

        :param points: iterable of (x, y) pixel coordinates.
        :return: list of (X, Y) field coordinates in meters.
        """
        pts_list = list(points)
        if not pts_list:
            return []

        arr = np.array(pts_list, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(arr, self.H)
        warped = warped.reshape(-1, 2)
        return [(float(X), float(Y)) for X, Y in warped]

