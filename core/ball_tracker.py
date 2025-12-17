import cv2
import numpy as np
from typing import Dict, List, Tuple
import torch
from scipy.signal import savgol_filter

class BallTracker:
    def __init__(self, confidence_threshold: float = 0.5, iou_threshold: float = 0.5):
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.ball_history = []
        self.velocity_history = []
        
    def track_ball(self, detections: List[Dict], frame: np.ndarray) -> Dict:
        """Enhanced ball tracking with multiple validation steps"""
        ball_candidates = self._filter_ball_detections(detections)
        
        if not ball_candidates:
            # Use motion prediction if no detection
            predicted_pos = self._predict_ball_position()
            if predicted_pos is not None:
                return self._create_ball_detection(predicted_pos)
            return None
            
        # Score candidates based on multiple criteria
        best_candidate = self._score_ball_candidates(ball_candidates, frame)
        
        # Update ball history
        self._update_ball_history(best_candidate)
        
        return best_candidate
    
    def _filter_ball_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter ball detections based on confidence and characteristics"""
        candidates = []
        
        for det in detections:
            if det['confidence'] < self.conf_threshold:
                continue
                
            # Calculate aspect ratio and area
            w, h = det['bbox'][2] - det['bbox'][0], det['bbox'][3] - det['bbox'][1]
            aspect_ratio = w / h
            area = w * h
            
            # Ball should be roughly circular and within reasonable size
            if 0.8 <= aspect_ratio <= 1.2 and 100 <= area <= 2500:
                candidates.append(det)
        
        return candidates
    
    def _score_ball_candidates(self, candidates: List[Dict], frame: np.ndarray) -> Dict:
        """Score candidates based on multiple criteria"""
        max_score = -1
        best_candidate = None
        
        for candidate in candidates:
            score = 0
            bbox = candidate['bbox']
            
            # Score based on confidence
            score += candidate['confidence'] * 0.3
            
            # Score based on circular shape
            circularity_score = self._calculate_circularity(frame, bbox)
            score += circularity_score * 0.3
            
            # Score based on motion consistency
            if self.ball_history:
                motion_score = self._calculate_motion_consistency(bbox)
                score += motion_score * 0.4
            
            if score > max_score:
                max_score = score
                best_candidate = candidate
        
        return best_candidate
    
    def _calculate_circularity(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate how circular the detected object is"""
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
            
        # Calculate circularity
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return min(circularity, 1.0)
    
    def _calculate_motion_consistency(self, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate how consistent the motion is with previous frames"""
        if len(self.ball_history) < 2:
            return 1.0
            
        current_center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
        prev_center = ((self.ball_history[-1][0] + self.ball_history[-1][2])/2,
                      (self.ball_history[-1][1] + self.ball_history[-1][3])/2)
        
        # Calculate current velocity
        current_velocity = (current_center[0] - prev_center[0],
                          current_center[1] - prev_center[1])
        
        if not self.velocity_history:
            return 1.0
            
        # Compare with average velocity
        avg_velocity = np.mean(self.velocity_history[-5:], axis=0)
        velocity_diff = np.linalg.norm(np.array(current_velocity) - avg_velocity)
        
        # Score based on velocity consistency
        return np.exp(-velocity_diff / 100)  # Exponential decay
    
    def _predict_ball_position(self) -> Tuple[float, float, float, float]:
        """Predict ball position based on motion history"""
        if len(self.ball_history) < 2:
            return None
            
        # Use last few positions to predict next position
        positions = np.array(self.ball_history[-5:])
        
        try:
            # Smooth the trajectories
            x = savgol_filter(positions[:, 0], min(5, len(positions)), 2)
            y = savgol_filter(positions[:, 1], min(5, len(positions)), 2)
            
            # Predict next position
            dx = x[-1] - x[-2]
            dy = y[-1] - y[-2]
            
            next_x = x[-1] + dx
            next_y = y[-1] + dy
            
            # Maintain same bbox size
            w = positions[-1, 2] - positions[-1, 0]
            h = positions[-1, 3] - positions[-1, 1]
            
            return (next_x, next_y, next_x + w, next_y + h)
        except:
            return None
    
    def _update_ball_history(self, detection: Dict):
        """Update ball position and velocity history"""
        if detection is None:
            return
            
        self.ball_history.append(detection['bbox'])
        if len(self.ball_history) > 30:  # Keep last 30 frames
            self.ball_history.pop(0)
            
        if len(self.ball_history) >= 2:
            current = np.array(self.ball_history[-1])
            prev = np.array(self.ball_history[-2])
            velocity = current - prev
            self.velocity_history.append(velocity)
            
            if len(self.velocity_history) > 30:
                self.velocity_history.pop(0) 