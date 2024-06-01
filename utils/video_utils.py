import cv2
from typing import List
import os
from datetime import datetime

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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed codec to 'mp4v' for MP4 format
    out = cv2.VideoWriter(video_path, fourcc, 24.0, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

def generate_output_filename(input_path, output_dir="output_videos"):
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{name}_{timestamp}{ext}")
    return output_path
