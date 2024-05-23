from typing import List, Tuple

def get_center_of_bbox(bbox: List[int]) -> Tuple[int, int]:
    """
    Calculate the center coordinates (x, y) of a bounding box.

    :param bbox: A list containing the bounding box coordinates in the format [x1, y1, x2, y2].
    :return: A tuple containing the x and y coordinates of the center.
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox: List[int]) -> int:
    """
    Calculate the width of a bounding box.

    :param bbox: A list containing the bounding box coordinates in the format [x1, y1, x2, y2].
    :return: The width of the bounding box.
    """
    x1, _, x2, _ = bbox
    return int(x2 - x1)

def measure_distance(p1, p2):
    """
    Calculate the distance between two points.

    :param p1: A tuple containing the x and y coordinates of the first point.
    :param p2: A tuple containing the x and y coordinates of the second point.
    :return: The distance between the two points.
    """
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0], p1[1]-p2[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)