"""
AudioRipple – demo launcher
===========================

Run three visualizers (spectrogram, ripple-field, helix) with any
combination of:

    • live microphone / interface
    • prerecorded WAV file
    • synthetic / heart-video excitations

Edit the `RUN_MODE` and feature flags at the top; no other changes
needed when you switch setups.
"""
from pathlib import Path
from typing import Dict, Union
import signal
import sys

# Third-party
import numpy as np
import librosa as lr
import qdarkstyle
from PyQt5 import QtCore, QtWidgets
from matplotlib import cm
from matplotlib.colors import Normalize
from loguru import logger

# Local modules
from audioviz.audio_processing.audio_processor import AudioProcessor
from audioviz.utils.audio_devices import select_devices, AudioDeviceDesktop
from audioviz.utils.guitar_profiles import GuitarProfile
from audioviz.visualization.spectrogram_visualizer import SpectrogramVisualizer
from audioviz.visualization.ripple_wave_visualizer import RippleWaveVisualizer
from audioviz.visualization.pitch_helix_visualizer import PitchHelixVisualizer

from audioviz.sources.audio import AudioExcitation
from audioviz.sources.heart_video import HeartVideoExcitation
from audioviz.sources.synthetic import SyntheticPointExcitation

# -----------------------------------------------------------------------------
# 1) HIGH-LEVEL SWITCHES
# -----------------------------------------------------------------------------
RUN_MODE = "live"        # "live" | "wav" | "headless"
FLAGS = dict(
    show_spectrogram=True,
    show_ripples=True,
    show_helix=False,

    use_audio_excitation=False,
    use_heart_video=True,
    use_synthetic=False,
)

HEART_VIDEO_PATH = Path("Data/GeneratedHearts/heart_mri.mp4")
WAV_PATH         = Path("data/test.wav")


# -----------------------------------------------------------------------------
# 2) GLOBAL PARAMS
# -----------------------------------------------------------------------------
SR_DEFAULT = 44100            # default when not streaming
N_FFT      = 256
WINDOW_MS  = 20
HOP_RATIO  = 1 / 4

RIPPLE_CONF = dict(
    plane_size_m=(0.30, 0.30),        # physical plane if needed
    resolution=(1440, 2560),          # (H, W)
    speed=340.0,
    amplitude=10.0,
    damping=0.90,
    use_gpu=True,
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

    # Spectrogram params
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

    # Ripple window & sources
    if FLAGS["show_ripples"]:
        ripple = RippleWaveVisualizer(**RIPPLE_CONF)
        ripple.setWindowTitle("Ripple Field")
        ripple.resize(600, 600)

        if FLAGS["use_audio_excitation"]:
            ripple.add_excitation_source(
                AudioExcitation(
                    name="Audio Ripple",
                    processor=processor,
                    position=(0.5, 0.5),
                    max_frequency=ripple.max_frequency,
                    amplitude=RIPPLE_CONF["amplitude"],
                    speed=RIPPLE_CONF["speed"],
                    resolution=RIPPLE_CONF["resolution"],
                    backend=ripple.backend,
                )
            )

        if FLAGS["use_heart_video"]:
            ripple.add_excitation_source(
                HeartVideoExcitation(
                    source=HEART_VIDEO_PATH,
                    resolution=RIPPLE_CONF["resolution"],
                    position=(0.3, 0.6),
                    backend=ripple.backend,
                    # audio_processor=processor,
                )
            )

        if FLAGS["use_synthetic"]:
            ripple.add_excitation_source(
                SyntheticPointExcitation(
                    name="Synthetic Ripple",
                    resolution=RIPPLE_CONF["resolution"],
                    position=(0.5, 0.5),
                    amplitude=10,
                    frequency=400,
                    speed=RIPPLE_CONF["speed"],
                    backend=ripple.backend,
                )
            )

        ripple.show()

    # Optional helix (Legacy)
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


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
