import cv2
from audioviz.sources.camera import CameraSource

def test_camera_basic():
    cam = CameraSource(camera_index=0, width=640, height=480)
    cam.start()

    try:
        while True:
            frame = cam.read()
            if frame is not None:
                print("Frame shape:", frame.shape)
                cv2.imshow("Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_basic()
