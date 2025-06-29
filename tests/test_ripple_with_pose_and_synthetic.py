import sys


import numpy as np
import cv2
from PyQt5 import QtWidgets


from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer

from audioviz.sources.camera import CameraSource
from audioviz.sources.pose.mediapipe_pose_source import MediaPipePoseExtractor
from audioviz.sources.pose.pose_graph_state import PoseGraphState
from audioviz.sources.synthetic import SyntheticPointExcitation

# -------------------- Parameters --------------------
RIPPLE_CONF = dict(
    plane_size_m=(1., 1.),
    dx=5e-3,
    speed=10.0,
    damping=0.90,
    use_gpu=True,
)

def draw_pose_graph(frame: np.ndarray, coords: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    coords_px = (coords * np.array([w, h])).astype(int)

    # Draw edges
    for i in range(len(coords)):
        for j in range(len(coords)):
            if adjacency[i, j]:
                cv2.line(frame, tuple(coords_px[i]), tuple(coords_px[j]), (0, 255, 0), 2)

    # Draw nodes
    for x, y in coords_px:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    return frame

def test_ripple_with_pose_and_synthetic() -> None:
    app = QtWidgets.QApplication(sys.argv)

    # --- Initialize visualizer ---
    ripple = RippleWaveVisualizer(**RIPPLE_CONF)
    ripple.setWindowTitle("Test Ripple with Pose + Synthetic")
    ripple.resize(800, 800)

    # --- Initialize pose graph ---
    camera = CameraSource(camera_index=0, width=640, height=480)
    camera.start()

    extractor = MediaPipePoseExtractor()
    pose_state = PoseGraphState(num_nodes=33, adjacency=extractor._get_static_adjacency(33))

    ripple.add_pose_graph(camera=camera, extractor=extractor, pose_graph_state=pose_state)

    # --- Add synthetic source ---
    synthetic = SyntheticPointExcitation(
        name="Synthetic Ripple",
        dx=RIPPLE_CONF["dx"],
        resolution=ripple.resolution,
        position=(0.5, 0.5),
        frequency=40,
        speed=RIPPLE_CONF["speed"],
        backend=ripple.backend,
    )
    ripple.add_excitation_source(synthetic)

    # --- Show visualizer ---
    ripple.show()

    # --- Pose debug loop ---
    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
        
            pose_data = extractor.extract(frame)
            coords = pose_data["coords"]
            adjacency = pose_data["adjacency"]
        
            if coords.shape[0] > 0:
                vis_frame = frame.copy()
                vis_frame = draw_pose_graph(vis_frame, coords, adjacency)
                cv2.imshow("Pose Graph Debug", vis_frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.stop()
        extractor.close()
        cv2.destroyAllWindows()
        print("✅ Pose debug loop cleanly stopped.")

    # --- Start Qt event loop after closing pose debug ---
    sys.exit(app.exec())

if __name__ == "__main__":
    test_ripple_with_pose_and_synthetic()
