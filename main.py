"""
AudioRipple – demo launcher
===========================

Run visualizers (spectrogram, ripple-field, helix) with any
combination of:

    • live microphone / interface
    • prerecorded WAV file
    • synthetic / heart-video excitations

Edit the `RUN_MODE` and FLAGS at the top; no other changes
needed when you switch setups.
"""
from pathlib import Path
from typing import Dict
import signal
import sys

import numpy as np
import librosa as lr
import qdarkstyle
from PyQt5 import QtCore, QtWidgets
from matplotlib import cm
from matplotlib.colors import Normalize
from loguru import logger

from audioviz.audio_processing.audio_processor import AudioProcessor
from audioviz.utils.audio_devices import select_devices
from audioviz.utils.guitar_profiles import GuitarProfile
from audioviz.visualization.spectrogram_visualizer import SpectrogramVisualizer
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer, RippleWaveVisualizer3D
from audioviz.visualization.pitch_helix_visualizer import PitchHelixVisualizer

from audioviz.sources.pose.mediapipe_pose_source import MediaPipePoseExtractor
from audioviz.sources.pose.pose_graph_state import PoseGraphState
from audioviz.sources.audio import AudioExcitation
from audioviz.sources.camera import CameraSource
from audioviz.sources.heart_video import HeartVideoExcitation
from audioviz.sources.synthetic import SyntheticPointExcitation

# -----------------------------------------------------------------------------
# 1) HIGH-LEVEL SWITCHES
# -----------------------------------------------------------------------------
RUN_MODE = "live"        # "live" | "wav" | "headless"
FLAGS = dict(
    show_spectrogram=False,
    show_ripples=True,
    show_helix=False,
    use_pose_graph=False,

    use_audio_excitation=True,
    use_heart_video=True,
    use_synthetic=True,

    # use_audio_excitation=True,
    # use_heart_video=False,
    # use_synthetic=True,
)

HEART_VIDEO_PATH = Path("Data/GeneratedHearts/test_e050_p002.avi")
WAV_PATH         = Path("data/test.wav")

# -----------------------------------------------------------------------------
# 2) GLOBAL PARAMS
# -----------------------------------------------------------------------------
SR_DEFAULT = 44100
N_FFT      = 256
WINDOW_MS  = 20
HOP_RATIO  = 1 / 4

USE_3D = True  # or False

RIPPLE_CONF = dict(
    plane_size_m=(50., 50.),
    # plane_size_m=(50., 25.),
    # plane_size_m=(100., 100.),
    dx=5e-2,
    speed=10.0,
    damping=0.90,
    use_gpu=True,
)
if RIPPLE_CONF["use_gpu"]:
    import cupy as cp
    BACKEND = cp
else:
    BACKEND = np
AUDIO_EXCITATION_CONF = dict(
    name="Audio Ripple",
    nominal_peak=1.0,
    position=(0.5, 0.5),
    decay_alpha=144.0,
    gain=0.0,
    speed=RIPPLE_CONF["speed"],
    dx=RIPPLE_CONF["dx"],
)

HEART_EXCITATION_CONF = dict(
    source=HEART_VIDEO_PATH,
    position=(0.5, 0.5),
    amplitude=1.0,
)

# -----------------------------------------------------------------------------
# 3) MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    # ---------------------------------------------------------------- Config
    if RUN_MODE == "live":
        config = select_devices(Path("outputs/audio_devices.json"))
        sr = config["samplerate"]
        audio_data = None
    elif RUN_MODE == "wav":
        audio_data, sr = lr.load(WAV_PATH, sr=None)
        config = dict(
            input_device_index=-1, input_channels=1,
            output_device_index=-1, output_channels=1,
            samplerate=sr,
        )
    else:  # headless
        sr, audio_data = SR_DEFAULT, None
        config = dict(
            input_device_index=-1, input_channels=1,
            output_device_index=-1, output_channels=1,
            samplerate=sr,
        )

    io_conf: Dict = {
        "is_streaming": RUN_MODE == "live",
        "input_device_index": config["input_device_index"],
        "input_channels": config["input_channels"],
        "output_device_index": config["output_device_index"],
        "output_channels": config["output_channels"],
        "io_blocksize": 4096,
    }

    win_len = 2 ** int(np.log2((WINDOW_MS / 1000) * sr))
    spec_params = dict(
        n_fft=N_FFT,
        hop_length=int(win_len * HOP_RATIO),
        n_mels=None,
        stft_window=lr.filters.get_window("hann", win_len),
        mel_spec_max=0.0,
    )

    # ---------------------------------------------------------------- Qt App
    app = QtWidgets.QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # ---------------------------------------------------------------- Audio IO
    processor = AudioProcessor(
        sr=int(sr),
        data=audio_data,
        num_samples_in_buffer=int(5 * sr),
        number_top_k_frequencies=N_FFT // 2,
        io_blocksize=io_conf["io_blocksize"],
        **{k: spec_params[k] for k in ("n_fft", "hop_length", "n_mels", "stft_window")},
        is_streaming=io_conf["is_streaming"],
        input_device_index=io_conf["input_device_index"],
        input_channels=io_conf["input_channels"] or 1,
        output_device_index=io_conf["output_device_index"],
        output_channels=io_conf["output_channels"] or 1,
    )

    # Pump raw input queue -> processor
    timer = QtCore.QTimer()
    timer.setInterval(int(io_conf["io_blocksize"] / sr * 1000))
    timer.timeout.connect(processor.process_pending_audio)
    timer.start()

    # ---------------------------------------------------------------- Visualizers
    if FLAGS["show_spectrogram"]:
        spectro = SpectrogramVisualizer(
            processor,
            cmap=cm.get_cmap("viridis"),
            norm=Normalize(vmin=-80, vmax=0),
            waveform_plot_duration=0.5,
        )
        spectro.setWindowTitle("Spectrogram")
        spectro.resize(800, 600)
        spectro.show()

    if FLAGS["show_ripples"]:

        if USE_3D:
            ripple = RippleWaveVisualizer3D(**RIPPLE_CONF)
        else:
            ripple = RippleWaveVisualizer(**RIPPLE_CONF)

        # ripple = RippleWaveVisualizer(**RIPPLE_CONF)
        ripple.setWindowTitle("Ripple Field")
        ripple.resize(800, 600)

        if FLAGS["use_pose_graph"]:
            camera = CameraSource(camera_index=0, width=640, height=480)
            camera.start()
            extractor = MediaPipePoseExtractor()
            pose_state = PoseGraphState(num_nodes=33, adjacency=extractor._get_static_adjacency(33))
            ripple.engine.add_pose_graph(camera=camera, extractor=extractor, pose_graph_state=pose_state)

        if FLAGS["use_audio_excitation"]:
            ripple.engine.add_source(
                AudioExcitation(
                    **AUDIO_EXCITATION_CONF,
                    processor=processor,
                    max_frequency=ripple.engine.max_frequency,
                    resolution=ripple.resolution,
                    backend=BACKEND,
                )
            )

        if FLAGS["use_heart_video"]:
            ripple.engine.add_source(
                HeartVideoExcitation(
                    **HEART_EXCITATION_CONF,

                    resolution=ripple.resolution,
                    backend=BACKEND,
                )
            )

        if FLAGS["use_synthetic"]:
            ripple.engine.add_source(
                SyntheticPointExcitation(
                    name="Synthetic Ripple",
                    dx=RIPPLE_CONF["dx"],
                    resolution=ripple.resolution,
                    position=(0.5, 0.5),
                    frequency=400,
                    speed=RIPPLE_CONF["speed"],
                    backend=BACKEND,
                )
            )

        ripple.show()
        ripple.toggle_controls()

    if FLAGS["show_helix"]:
        helix = PitchHelixVisualizer(
            processor,
            guitar_profile=GuitarProfile(
                open_strings=[82.41, 110, 146.83, 196, 246.94, 329.63],
                num_frets=22,
            ),
        )
        helix.resize(800, 600)
        helix.show()

    # ---------------------------------------------------------------- Start IO
    if not processor.start():
        logger.error("Audio start failed.")
        return

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app.aboutToQuit.connect(processor.stop)

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        processor.stop()


if __name__ == "__main__":
    main()
