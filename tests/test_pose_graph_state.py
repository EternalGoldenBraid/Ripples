from typing import Optional

import cv2
import numpy as np
from audioviz.sources.camera import CameraSource
from audioviz.sources.pose.mediapipe_pose_source import MediaPipePoseExtractor
from audioviz.sources.pose.pose_graph_state import PoseGraphState

class TestPoseGraphState:
    def test_pose_state_visualize(self) -> None:
        cam = CameraSource(camera_index=0, width=640, height=480)
        extractor = MediaPipePoseExtractor()

        cam.start()

        dt = 1 / 30.0  # assume ~30 FPS camera
        state_obj: Optional[PoseGraphState] = None

        try:
            while True:
                frame = cam.read()
                if frame is not None:
                    pose_data = extractor.extract(frame)
                    coords = pose_data['coords']
                    adjacency = pose_data['adjacency']

                    if coords.shape[0] > 0:
                        if state_obj is None:
                            state_obj = PoseGraphState(coords.shape[0], adjacency, velocity_smoothing_alpha=0.8)
                        
                        state_obj.update(coords, dt)

                        positions = state_obj.get_positions()
                        velocities = state_obj.get_velocities()
                        accelerations = state_obj.get_accelerations()

                        frame_vis = frame.copy()

                        coords_px = (positions * np.array([frame.shape[1], frame.shape[0]])).astype(int)
                        vel_norm = np.linalg.norm(velocities, axis=1)
                        acc_norm = np.linalg.norm(accelerations, axis=1)

                        # Map velocity norm to color (blue to red)
                        colors = np.clip(vel_norm / (vel_norm.max() + 1e-6), 0, 1)
                        colors_bgr = [(int(255 * (1 - c)), 0, int(255 * c)) for c in colors]

                        # Map acceleration to node size
                        sizes = np.clip(5 + 20 * (acc_norm / (acc_norm.max() + 1e-6)), 5, 25)

                        for idx, (x, y) in enumerate(coords_px):
                            color = colors_bgr[idx]
                            radius = int(sizes[idx])
                            cv2.circle(frame_vis, (x, y), radius, color, -1)

                        cv2.imshow("PoseGraphState Visualization", frame_vis)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            extractor.close()
            cam.stop()
            cv2.destroyAllWindows()
            print("✅ Cleanly stopped.")
    

if __name__ == "__main__":
    tester = TestPoseGraphState()
    tester.test_pose_state_visualize()
