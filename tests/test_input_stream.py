# import pytest
import sounddevice as sd
import numpy as np

from audioviz.utils.audio_devices import (
    AudioDeviceMSI,
    AudioDeviceDesktop,
)

class TestAudioStreams:
    def setUp(self):
        """Set up the necessary attributes for the test."""
        self.sr = 44100  # Sampling rate
        self.blocksize = 1024  # Number of frames per callback


        self.device_enum = AudioDeviceDesktop
        
        # self.output_device_index: int = self.device_enum.SPEAKERS.value
        self.output_device_index: int = self.device_enum.SCARLETT_SOLO_USB.value
        self.output_channels: int = 2
        self.input_device_index: int = self.device_enum.SCARLETT_SOLO_USB.value
        self.input_channels: int = 2

        assert self.output_device_index == 8, "Output device index is not 4"
        assert self.input_device_index == 8, "Input device index is not 8"

        # Allocate a buffer to hold one block of audio data
        self.channels = max(self.input_channels, self.output_channels)
        self.audio_buffer = np.zeros((self.blocksize, self.channels))

    def test_audio_stream_with_delay(self):
        """Test input and output audio streams with delay."""
        
        def audio_input_callback(indata, frames, time, status):
            """Callback for input stream (recording)."""
            if status:
                print(f"Input Stream Error: {status}")

            self.audio_buffer[:frames] = indata[:frames]
            print(f"Indata: {indata.shape}")

        def audio_output_callback(outdata, frames, time, status):
            """Callback for output stream (playback)."""
            if status:
                print(f"Output Stream Error: {status}")

            # Play back the data from the buffer with a slight delay
            outdata[:] = self.audio_buffer[-frames:]

        # Create an input stream (for recording)
        input_stream = sd.InputStream(
            samplerate=self.sr, channels=self.input_channels,
            blocksize=self.blocksize, callback=audio_input_callback,
            device=self.input_device_index
        )

        # Create an output stream (for playback)
        output_stream = sd.OutputStream(
            samplerate=self.sr, channels=self.output_channels,
            blocksize=self.blocksize, callback=audio_output_callback,
            device=self.output_device_index
        )

        try:
            # Start both streams
            input_stream.start()
            output_stream.start()

            # Record and play audio with a slight delay
            sd.sleep(15000)  # 2 seconds to record and playback the input

        except Exception as e:
            print(f"Audio stream failed with error: {e}")

        finally:
            # Stop the streams
            output_stream.stop()
            input_stream.stop()

if __name__ == "__main__":
    test = TestAudioStreams()
    test.setUp()
    test.test_audio_stream_with_delay()
