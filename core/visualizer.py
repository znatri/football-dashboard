import cv2
import numpy as np
from typing import Dict, List, Tuple

class MatchVisualizer:
    def __init__(self):
        self.team_colors = {
            1: (255, 50, 50),   # Red for team 1
            2: (50, 50, 255)    # Blue for team 2
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def draw_annotations(self, frame: np.ndarray, tracks: Dict, 
                        ball_track: Dict, team_possession: int) -> np.ndarray:
        """Draw enhanced annotations on the frame"""
        annotated_frame = frame.copy()
        
        # Draw field overlay
        annotated_frame = self._draw_field_overlay(annotated_frame)
        
        # Draw players
        for player_id, track in tracks['players'].items():
            annotated_frame = self._draw_player(
                annotated_frame, 
                track['bbox'], 
                player_id,
                track['team'],
                track.get('speed_kmph', 0),
                track.get('has_ball', False)
            )
        
        # Draw ball with enhanced effects
        if ball_track:
            annotated_frame = self._draw_ball(
                annotated_frame,
                ball_track['bbox'],
                ball_track.get('velocity', (0, 0))
            )
        
        # Draw possession indicator
        self._draw_possession_indicator(annotated_frame, team_possession)
        
        return annotated_frame
    
    def _draw_field_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw semi-transparent field overlay with zones"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw zones
        cv2.line(overlay, (w//3, 0), (w//3, h), (255, 255, 255), 1)
        cv2.line(overlay, (2*w//3, 0), (2*w//3, h), (255, 255, 255), 1)
        
        # Add transparency
        return cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
    
    def _draw_player(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                    player_id: int, team: int, speed: float, has_ball: bool) -> np.ndarray:
        """Draw enhanced player annotation"""
        x1, y1, x2, y2 = map(int, bbox)
        color = self.team_colors[team]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw player info background
        info_bg = np.zeros((60, 100, 3), dtype=np.uint8)
        cv2.rectangle(info_bg, (0, 0), (100, 60), color, -1)
        
        # Add player info
        cv2.putText(info_bg, f"P{player_id}", (5, 20), self.font, 0.5, (255, 255, 255), 1)
        cv2.putText(info_bg, f"{speed:.1f} km/h", (5, 40), self.font, 0.5, (255, 255, 255), 1)
        
        # Add ball indicator
        if has_ball:
            cv2.circle(info_bg, (80, 30), 10, (255, 255, 255), -1)
        
        # Blend info box onto frame
        y_offset = max(0, y1 - 70)
        x_offset = max(0, x1)
        frame[y_offset:y_offset+60, x_offset:x_offset+100] = cv2.addWeighted(
            frame[y_offset:y_offset+60, x_offset:x_offset+100],
            0.5,
            info_bg,
            0.5,
            0
        )
        
        return frame
    
    def _draw_ball(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                  velocity: Tuple[float, float]) -> np.ndarray:
        """Draw enhanced ball annotation"""
        x1, y1, x2, y2 = map(int, bbox)
        center = (int((x1 + x2)/2), int((y1 + y2)/2))
        
        # Draw motion blur effect
        if abs(velocity[0]) > 1 or abs(velocity[1]) > 1:
            alpha = 0.3
            for i in range(1, 4):
                prev_pos = (
                    int(center[0] - i * velocity[0]),
                    int(center[1] - i * velocity[1])
                )
                cv2.circle(frame, prev_pos, 5, (255, 255, 255), -1)
                frame = cv2.addWeighted(frame, 1-alpha, frame, alpha, 0)
        
        # Draw ball
        cv2.circle(frame, center, 5, (0, 255, 255), -1)
        cv2.circle(frame, center, 7, (0, 255, 255), 1)
        
        return frame
    
    def _draw_possession_indicator(self, frame: np.ndarray, team_possession: int):
        """Draw team possession indicator"""
        h, w = frame.shape[:2]
        
        if team_possession > 0:
            color = self.team_colors[team_possession]
            
            # Draw possession indicator
            cv2.rectangle(frame, (w-200, 20), (w-20, 60), color, -1)
            cv2.putText(frame, f"Team {team_possession} Possession",
                       (w-190, 45), self.font, 0.6, (255, 255, 255), 2) 