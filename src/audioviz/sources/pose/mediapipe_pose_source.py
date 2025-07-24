import numpy as np
import mediapipe as mp
from typing import Dict
from .base import PoseGraphExtractor

class MediaPipePoseExtractor(PoseGraphExtractor):
    def __init__(self,
                 static_image_mode: bool = False,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        results = self.pose.process(frame[..., ::-1])  # Convert BGR to RGB

        if not results.pose_landmarks:
            return {'coords': np.zeros((0, 2), dtype=np.float32),
                    'adjacency': np.zeros((0, 0), dtype=np.float32)}

        landmarks = results.pose_landmarks.landmark
        coords = np.array([[l.x, l.y] for l in landmarks], dtype=np.float32)  # normalized

        adjacency = self._get_static_adjacency(len(coords))
        return {'coords': coords, 'adjacency': adjacency}

    def _get_static_adjacency(self, num_nodes: int) -> np.ndarray:
        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        # Example connections (from MediaPipe official skeleton)
        edges = [
            (11, 13), (13, 15), (12, 14), (14, 16),  # arms
            (11, 12), (23, 24),                      # shoulders and hips
            (11, 23), (12, 24),                      # torso sides
            (23, 25), (25, 27), (27, 29), (29, 31), # left leg
            (24, 26), (26, 28), (28, 30), (30, 32), # right leg
            (15, 21), (16, 22),                      # hands to fingers (approx.)
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10) # head connections (approx.)
        ]
    
        for i, j in edges:
            if i < num_nodes and j < num_nodes:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0  # symmetric
    
        return adjacency

    def close(self) -> None:
        self.pose.close()
