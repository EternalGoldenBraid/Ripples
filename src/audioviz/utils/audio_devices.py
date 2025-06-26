from typing import Dict, Any, Union
from enum import Enum
from json import dumps, loads
from pathlib import Path

import sounddevice as sd

class AudioDeviceMSI(Enum):
    SCARLETT_SOLO_USB = 0
    ALC298_ANALOG = 1
    HDMI_0 = 2
    HDMI_1 = 3
    HDMI_2 = 4
    SYSDEFAULT = 5
    FRONT = 6
    SURROUND40 = 7
    DMIX = 8


class AudioDeviceDesktop(Enum):
    SAMSUNG = 0
    SPEAKERS = 4
    ALCS1200A_ANALOG = 5
    ALCS1200A_DIGITAL = 6
    ALCS1200A_ALT_ANALOG = 7
    SCARLETT_SOLO_USB = 8

DesktopDevicesConfigs: Dict[str, Dict[str, Any]] = {
    "SCARLETT_SOLO_USB": {
        "channels": 2,
        "samplerate": 44100,
        "channel_metadata": {"in": "1: Mic, 2: Guitar"},
    }
}


def list_devices():
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        io_flag = []
        if dev['max_input_channels'] > 0:
            io_flag.append("IN")
        if dev['max_output_channels'] > 0:
            io_flag.append("OUT")
        print(f"[{idx}] {dev['name']} ({', '.join(io_flag)})")

    # Null device is always available
    print(f"[{idx+1}] Null Device (no audio input/output)")
    return devices

def prompt_for_device(devices, kind: str = "input"):
    while True:
        try:
            index = int(input(f"Select {kind} device index: "))

            if index == len(devices):
                print("Using Null Device (no audio input/output)")
                index = -1
                return index, {
                    "name": "Null Device",
                    "max_input_channels": 0, "max_output_channels": 0,
                    "default_samplerate": 44100,
                }

            device = devices[index]
            if kind == "input" and device['max_input_channels'] == 0:
                print("Selected device has no input channels.")
                continue
            if kind == "output" and device['max_output_channels'] == 0:
                print("Selected device has no output channels.")
                continue
            return index, device
        except (ValueError, IndexError):
            print("Invalid index. Try again.")

def select_devices(config_file: Path = Path("config.json")):
    if config_file.exists():
        print(f"Reading configuration from {config_file}")
        with open(config_file, "r") as f:
            return loads(f.read())
    else:
        print(f"Configuration file not found. Creating new configuration at {config_file}")
        if not config_file.parent.exists():
            config_file.parent.mkdir(parents=True, exist_ok=True)

    print("=== Available Audio Devices ===")
    devices = list_devices()
    
    print("\n--- Select Input Device ---")
    input_index, input_dev = prompt_for_device(devices, kind="input")
    
    print("\n--- Select Output Device ---")
    output_index, output_dev = prompt_for_device(devices, kind="output")
    
    config: Dict[str, Union[int, str]] = {
        "input_device_index": input_index,
        "output_device_index": output_index,
        "input_channels": input_dev['max_input_channels'],
        "output_channels": output_dev['max_output_channels'],
        "samplerate": int(input_dev['default_samplerate'])  # assume same SR
    }

    with open(config_file, "w") as f:
        f.write(dumps(config, indent=4))

    return config
