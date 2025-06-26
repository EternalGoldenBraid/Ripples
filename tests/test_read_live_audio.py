from pathlib import Path
from typing import Dict, Union
from json import dumps, loads

import sounddevice as sd

from audioviz.utils.audio_devices import (
    AudioDeviceMSI,
    AudioDeviceDesktop,
)
from audioviz.utils.audio_devices import select_devices

class TestAudioDeviceSetup:

    def test_read_live_audio(self):
        # Set the device index (Scarlett Solo is now index 0)
        # device_enum = AudioDeviceMSI
        device_enum = AudioDeviceDesktop
        
        device_index: int = device_enum.SCARLETT_SOLO_USB.value
        channels: int = 2  # Assuming stereo input (you can change it depending on your setup)
        
        # Set the sample rate
        samplerate = 44100
        
        print(sd.query_devices())

    def test_select_devices_prompt(self):
        config = select_devices(config_file=Path("outputs/audio_devices.json"))
        print("\n🎧 Selected Configuration:")
        print(config)
        

if __name__ == "__main__":
    test = TestAudioDeviceSetup()

    # test.test_read_live_audio()

    test.test_select_devices_prompt()
