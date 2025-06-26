# Audio Ripple 🌊

**Audio Ripple** is an immersive audio-reactive visualizer for live performance, interactive installations, and creative experimentation.  
It converts sound (or any excitation plug-in) into dynamic ripple waves that propagate across a 2-D plane.

---

## ✨ Highlights

| Feature | Details |
|---------|---------|
| 🔊 **Real-time audio** | Microphone / any `sounddevice` input |
| 🛠 **Modular sources** | Audio, synthetic tones, soon: video & heart-pulse |
| 🚀 **GPU acceleration** | Optional CuPy backend |
| 🎚 **Live controls** | Damping, speed, amplitude, decay, per-source sliders |
| 🖥 **Qt / PyQtGraph GUI** | Responsive, resizeable window |
| 🧩 **Extensible** | Add new sources by subclassing `ExcitationSourceBase` |

---

## 🔧 Installation

### 1. Clone

```bash
git clone https://github.com/<you>/audio-ripple.git
cd audio-ripple

### 2. Set-up environment

> Preferred (Pixi)
```bash
# Installs all deps from pixi.toml
pixi install
# Run the main app
pixi run python main.py
```

> Alternative (pip)

```bash
python -m venv .venv
source .venv/bin/activate            # or .venv\Scripts\activate on Windows
pip install -r requirements.txt      # Generate from pixi.toml
python main.py
```

Adjust runtime settings (resolution, plane size, GPU use) in main.py.

## 🎛 Interactive Controls
| Slider        | Effect                           |
| ------------- | -------------------------------- |
| **Damping**   | Energy loss per step             |
| **Decay α**   | Spatial fall-off of each source  |
| **Amplitude** | Excitation strength (per-source) |
| **Speed**     | Wave propagation speed (m/s)     |

Additional sliders appear automatically for each custom source.

## 🧠 Project Structure

```
src/audioviz
├── audio_processing/
│   └── audio_processor.py
├── sources/             # <— plug-in exciters live here
│   ├── audio.py
│   ├── synthetic.py
│   ├── image_pulse.py   # (WIP)
│   └── base.py
├── visualization/
│   ├── ripple_wave_visualizer.py
│   ├── spectrogram_visualizer.py
│   └── visualizer_base.py
└── utils/
    ├── audio_devices.py
    └── signal_processing.py
main.py
```

## 📚 Documentation
Generate local API docs with pdoc:
```bash
PYTHONPATH=src pdoc audioviz --html --output-dir docs --force
xdg-open docs/audioviz/index.html   # or `open` / `start` on macOS / Windows
```

## 🤝 Contributing
We love PRs!
See `CONTRIBUTING.md` for setup, coding style, and how to add new excitation sources (e.g., `HeartPulseExcitation`).

## 🎥 Demo & Gallery (coming soon)

Place demo GIFs / MP4s in assets/ and embed them here.
A phone-recorded clip is perfectly fine for now.

## 📄 License

MIT — see LICENSE.txt.

## 📫 TODO:
- [ ] Add more excitation sources (e.g., video, heart pulse)
- [ ] Write CONTRIBUTING.md
- [ ] Clean `main.py` and `*.yaml` or structured configs.
