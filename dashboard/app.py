import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import tempfile
import sys
from dataclasses import dataclass
import argparse
import yaml
from pathlib import Path
import subprocess
import json
from ultralytics import YOLO
import torch.hub
import requests
from typing import Dict, Any
import plotly.graph_objects as go
import time

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import process_video_pipeline  # Import the main processing function

@dataclass
class Config:
    """Video processing configuration"""
    video_path: str
    output_path: str = "output.mp4"
    verbose: bool = True
    model_path: str = "models/best.pt"
    save: bool = False

class Logger:
    """Streamlit logger"""
    def __init__(self, placeholder):
        self.out = placeholder

    def debug(self, msg): self.out.write(f"Status: {msg}")
    def info(self, msg): self.out.write(f"Info: {msg}")
    def warning(self, msg): self.out.warning(msg)
    def error(self, msg): self.out.error(msg)

def main():
    st.set_page_config(page_title="Football Analytics", layout="wide")
    st.title("Football Analytics")
    
    mode = st.sidebar.radio("Mode", ["Analysis", "Training"])
    if mode == "Analysis":
        analyze_video()
    else:
        train_model()

def analyze_video():
    """Video analysis view"""
    video = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi'])
    model = st.sidebar.text_input("Model Path", value="models/best.pt")
    
    if not video:
        return
        
    # Save video temporarily
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp.write(video.read())
    path = temp.name
    
    try:
        # Setup UI
        vid_col, stats_col = st.columns([2, 1])
        
        with vid_col:
            st.subheader("Video")
            frame_view = st.empty()
            progress = st.progress(0)

        with stats_col:
            st.subheader("Stats")
            stats_view = st.container()
            with stats_view:
                possession = st.empty()
                players = st.empty()

        # Process video
        cfg = Config(video_path=path, model_path=model)
        log = Logger(st.empty())
        
        def update(frame_idx, total, frame, tracks, ball_control):
            pct = int((frame_idx + 1) / total * 100)
            progress.progress(pct)
            frame_view.image(frame, channels="BGR", use_column_width=True)
            update_stats(possession, players, tracks, frame_idx, ball_control)

        process_video_pipeline(cfg, log, update)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        if os.path.exists(path):
            os.unlink(path)

def update_stats(possession_view, players_view, tracks, frame_idx, ball_control):
    """Update stats display"""
    try:
        # Show possession
        if ball_control and len(ball_control) > 0:
            t1_pct = (np.sum(np.array(ball_control) == 1) / len(ball_control)) * 100
            t2_pct = (np.sum(np.array(ball_control) == 2) / len(ball_control)) * 100
            
            possession_view.markdown(f"""
                ### Possession
                - Team 1: {t1_pct:.1f}%
                - Team 2: {t2_pct:.1f}%
            """)

        # Show player speeds
        if tracks and isinstance(tracks, dict) and "players" in tracks and frame_idx in tracks["players"]:
            speeds = []
            for pid, track in tracks["players"][frame_idx].items():
                if isinstance(track, dict) and "speed_kmph" in track:
                    speeds.append(f"Player {pid}: {track['speed_kmph']:.1f} km/h")
            
            if speeds:
                players_view.markdown("### Player Speeds\n" + "\n".join(speeds))
                
    except Exception as e:
        st.error(f"Stats error: {str(e)}")

def train_model():
    """Model training view"""
    st.subheader("Model Training")
    
    # Settings
    data_path = st.text_input("Dataset Path", "data/")
    epochs = st.number_input("Epochs", 1, 1000, 100)
    batch = st.number_input("Batch Size", 1, 128, 16)
    device = st.selectbox("Device", ["cuda", "cpu"])
    
    if st.button("Train"):
        progress = st.progress(0)
        status = st.empty()
        
        # Demo progress
        for i in range(epochs):
            pct = (i + 1) / epochs
            progress.progress(pct)
            status.text(f"Epoch {i+1}/{epochs}")
            time.sleep(0.1)
        
        st.success("Training complete!")

if __name__ == "__main__":
    main() 