import cv2
from typing import List

def read_video(video_path):
    """
    Read a video and return a list of frames.

    :param video_path: path to the video
    :param return: list of frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
     
    cap.release()
    return frames

def save_video(frames: List, video_path: str) -> None:
    """
    Save a list of frames as a video.

    :param frames: list of frames
    :param video_path: path to save the video
    """
    if not frames:
        raise ValueError("No frames to save")

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, 24.0, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()