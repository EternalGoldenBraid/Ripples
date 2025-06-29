import cv2
import numpy as np
from typing import Optional

from audioviz.sources.camera import CameraSource
from audioviz.sources.pose.mediapipe_pose_source import MediaPipePoseExtractor


class TestPoseExtractor:
    def test_construct(self) -> None:
        extractor = MediaPipePoseExtractor()
        assert extractor is not None
        print("✅ Pose extractor constructed successfully.")
        extractor.close()

    def test_camera_extraction_and_visualize(self) -> None:
        cam = CameraSource(camera_index=0, width=640, height=480)
        extractor = MediaPipePoseExtractor()

        cam.start()

        try:
            while True:
                frame: Optional[np.ndarray] = cam.read()
                if frame is not None:
                    pose_data = extractor.extract(frame)
                    coords = pose_data['coords']
                    adjacency = pose_data['adjacency']

                    frame_vis = frame.copy()

                    if coords.size > 0:
                        coords_px = (coords * np.array([frame.shape[1], frame.shape[0]])).astype(int)

                        # Draw edges first
                        for i in range(adjacency.shape[0]):
                            for j in range(i + 1, adjacency.shape[1]):
                                if adjacency[i, j] > 0:
                                    pt1 = tuple(coords_px[i])
                                    pt2 = tuple(coords_px[j])
                                    cv2.line(frame_vis, pt1, pt2, (255, 0, 0), 2)
                                    print(f"Drawing edge between {pt1} and {pt2}")

                        # Draw keypoints
                        for x, y in coords_px:
                            cv2.circle(frame_vis, (x, y), 5, (0, 255, 0), -1)

                    cv2.imshow("Pose Graph (with adjacency)", frame_vis)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            extractor.close()
            cam.stop()
            cv2.destroyAllWindows()
            print("✅ Camera and extractor stopped cleanly.")


if __name__ == "__main__":
    tester = TestPoseExtractor()
    tester.test_construct()
    tester.test_camera_extraction_and_visualize()
