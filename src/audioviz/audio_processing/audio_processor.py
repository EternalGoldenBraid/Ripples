from collections import deque
from typing import Optional, Union, Tuple, List

import numpy as np
import sounddevice as sd
from functools import partial
import librosa as lr
from loguru import logger

from audioviz.utils.signal_processing import perceptual_soft_threshold

def get_stft_spectrogram(segment: np.ndarray, hop_length: int,
                         stft_window: np.ndarray, n_fft: int) -> np.ndarray:
    spectrogram = np.abs(
        lr.stft(segment, n_fft=stft_window.shape[0],
                hop_length=hop_length, window=stft_window)[:n_fft // 2 + 1, :]
    )
    return spectrogram

def get_mel_spectrogram(segment: np.ndarray,
                        stft_window: np.ndarray, n_fft: int,
                        hop_length: int, n_mels: int, sr: int) -> np.ndarray:
    spectrogram = lr.feature.melspectrogram(
        y=segment, sr=sr, window=stft_window,
        n_fft=stft_window.shape[0], hop_length=hop_length, n_mels=n_mels
    )
    return spectrogram

class AudioProcessor:
    def __init__(self,
                 sr: int,
                 n_fft: int,
                 hop_length: int,
                 num_samples_in_buffer: int,
                 stft_window: np.ndarray,
                 io_blocksize: int,
                 number_top_k_frequencies: int,
                 n_mels: Optional[int] = None,
                 is_streaming: bool = False,
                 input_device_index: Optional[int] = None,
                 input_channels: int = 1,
                 output_device_index: Optional[int] = None,
                 output_channels: int = 1,
                 data: Optional[np.ndarray] = None):
        """
        TODO 
        - [ ] Channel wise spectrogram config (window size etc...)
        """


        self.sr: float = sr
        self.n_fft: int = n_fft
        self.hop_length: int = hop_length
        self.n_mels: int = n_mels
        self.num_samples_in_buffer: int = num_samples_in_buffer
        self.stft_window: Union[str, tuple, np.ndarray] = stft_window
        self.is_streaming: bool = is_streaming
        self.io_blocksize: int = io_blocksize
        self.data = data

        self.input_channels: int = input_channels
        self.output_channels: int = output_channels

        self.audio_buffer: np.ndarray = np.zeros(
            (num_samples_in_buffer, input_channels), dtype=np.float32
        )
        self.n_spec_bins: int = n_mels if n_mels is not None else n_fft // 2 + 1
        n_spec_frames: int = num_samples_in_buffer // hop_length
        # self.spectrogram_buffer: np.ndarray = np.zeros(
        #     (self.n_spec_bins, n_spec_frames), dtype=np.float32
        # )
        self.spectrogram_buffers: List[np.ndarray] = [
            np.zeros((self.n_spec_bins, n_spec_frames), dtype=np.float32)
            for _ in range(self.input_channels)
        ]

        self.snapshot_queue: deque[Tuple[np.ndarray, List[np.ndarray]]] = deque(maxlen=5)
        self.num_top_frequencies: int = 3
        # self.current_top_k_frequencies: list[float] = [None] * self.num_top_frequencies
        self.current_top_k_frequencies: np.ndarray = np.zeros(
            (self.input_channels, self.num_top_frequencies), dtype=np.float32
        )
        self.current_top_k_energies: np.ndarray = np.zeros_like(
            self.current_top_k_frequencies)

        self.freq_bins = np.fft.rfftfreq(self.n_fft, d=1/self.sr)

        self.raw_input_queue: deque[np.ndarray] = deque(maxlen=32)

        if n_mels is None:
            self.compute_spectrogram = partial(
                get_stft_spectrogram,
                stft_window=stft_window,
                n_fft=n_fft,
                hop_length=hop_length
            )
        else:
            self.compute_spectrogram = partial(
                get_mel_spectrogram,
                stft_window=stft_window,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                sr=sr
            )

        self.current_time: float = 0.0  # in seconds
        self.frame_counter: int = 0
        self.input_overflow_count: int = 0

        if output_device_index != -1:
            logger.info(f"Using output device index: {output_device_index}")
            self.stream = sd.OutputStream(
                samplerate=sr,
                device=output_device_index,
                channels=output_channels,
                callback=self.audio_output_callback,
                blocksize=io_blocksize
            )

        if self.is_streaming:
            if input_device_index != -1:
                logger.info(f"Using input device index: {input_device_index}")
                self.input_stream = sd.InputStream(
                    device=input_device_index,
                    channels=input_channels,
                    samplerate=sr,
                    callback=self.audio_input_callback,
                    blocksize=io_blocksize,
                    dtype='float32', latency='low',
                )

    def start(self):

        if not hasattr(self, 'stream') and not hasattr(self, 'input_stream'):
            logger.error("No input nor output audio streams initialized. Please check your configuration.")
            return False

        if not hasattr(self, 'stream') or self.stream is None:
            logger.info("Output stream is not initialized.")
        else:
            self.stream.start()
        if self.is_streaming and hasattr(self, 'input_stream'):
            self.input_stream.start()

        return True

    def get_smoothed_top_k_peak_frequency(
            self,
            k: int = 3,
            window_frames: int = 3
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return top-k frequency values AND their corresponding energies for each selected channel.
    
        Returns:
            Tuple[freqs, energies], both shape (C, k)
        """
    
        channel_indices = range(len(self.spectrogram_buffers)) # Taking all channels
        stacked = np.stack(
            [self.spectrogram_buffers[ch][:, -window_frames:] for ch in channel_indices],
            axis=0  # shape: (C, F, T)
        )
        averaged = np.mean(stacked, axis=2)  # shape: (C, F)
    
        freq_bins = np.fft.rfftfreq(self.n_fft, d=1/self.sr)
        top_k_idxs = np.argsort(averaged, axis=1)[:, -k:]  # shape: (C, k)
    
        freqs = freq_bins[top_k_idxs]              # shape: (C, k)
        energies = np.take_along_axis(averaged, top_k_idxs, axis=1)  # shape: (C, k)
        return top_k_idxs, freqs, energies

    def get_smoothed_top_k_peak_frequency_(self, 
            channel_idx: int = 1,
            k:int = 3, window_frames: int = 3) -> Optional[Tuple[List[int], List[float]]]:
        """
        Return dominant frequency smoothed over last `window_frames` spectrogram frames.
        """
        if self.spectrogram_buffers is None:
            return None

        # Average over last `window_frames` frames
        averaged_frame = np.mean(self.spectrogram_buffers[channel_idx][:, -window_frames:], axis=1)

        freq_bins = np.fft.rfftfreq(self.n_fft, d=1/self.sr)
        top_k_idxs = np.argsort(averaged_frame)[-k:]
        return top_k_idxs.tolist(), freq_bins[top_k_idxs].tolist()

    def update_spectrogram_buffer(self, indata: np.ndarray) -> None:
        for channel_idx in range(indata.shape[1]):
            segment = indata[:, channel_idx]
            # Discard the symmetric half
            spectrogram = self.compute_spectrogram(segment=segment)

            # # Filter out low-energy frames
            # spectrogram[spectrogram < 0.3] = 0.0
            # from audioviz.utils.signal_processing import linear_soft_threshold
            # spectrogram = linear_soft_threshold(spectrogram, thresh=0.1, fade_width=0.1)

            # spectrogram, _ = perceptual_soft_threshold(
            #     indata, alpha=1.0, beta=0.2
            # )

            frames_spec = spectrogram.shape[1]
            self.spectrogram_buffers[channel_idx] = np.roll(
                self.spectrogram_buffers[channel_idx], -frames_spec, axis=1
            )
            self.spectrogram_buffers[channel_idx][:, -frames_spec:] = spectrogram

    def process_audio(self, indata: np.ndarray):
        frames = indata.shape[0]

        self.audio_buffer = np.roll(self.audio_buffer, -frames, axis=0)
        self.audio_buffer[-frames:] = indata

        self.update_spectrogram_buffer(indata)

        idxs_, self.current_top_k_frequencies[:], self.current_top_k_energies[:] = \
            self.get_smoothed_top_k_peak_frequency(
                # window_frames=10, k=self.num_top_frequencies, channel_idx=0)
                window_frames=10, k=self.num_top_frequencies)

        self.snapshot_queue.append((
            self.audio_buffer.copy(),
            [spec.copy() for spec in self.spectrogram_buffers]
        ))

        self.frame_counter += 1

    def process_pending_audio(self):
        while self.raw_input_queue:
            audio_chunk = self.raw_input_queue.popleft()
            self.process_audio(audio_chunk)


    def audio_input_callback(self, indata: np.ndarray, frames: int, time, status):

        # if status and self.frame_counter % 10 == 0:
        #     logger.warning(f"Input stream error: {status}")
        if status:
            self.input_overflow_count += 1
            if self.input_overflow_count % 10 == 0:
                logger.warning(f"Input overflows: {self.input_overflow_count}, {status}")


        # Push a copy into a fast queue
        self.raw_input_queue.append(indata.copy())

    def audio_output_callback(self, outdata, frames, time, status):
        if status and self.frame_counter % 10 == 0:
            logger.warning(f"Output stream error: {status}")

        # For file playback, we can use the data directly
        # if self.data is not None and not self.is_streaming:
        #     start_idx = int(self.current_time * self.sr)
        #     end_idx = start_idx + frames
        #     if end_idx > len(self.data):
        #         indata = np.zeros(frames, dtype=np.float32)
        #         indata[:len(self.data[start_idx:])] = self.data[start_idx:]
        #         raise sd.CallbackStop()
        #     else:
        #         indata = self.data[start_idx:end_idx]
        #
        #     if self.audio_buffer.shape[1] > 1:
        #         indata = np.tile(indata[:, np.newaxis], (1, self.audio_buffer.shape[1]))
        #
        #     self.audio_buffer[-frames:] = indata

        outdata[:] = self.audio_buffer[-frames:]
        self.current_time += frames / self.sr

    def get_channel_spectrogram(self, channel_idx: int) -> np.ndarray:
        """Return the spectrogram of the specified channel."""
        if 0 <= channel_idx < len(self.spectrogram_buffers):
            return self.snapshot_queue[-1][1][channel_idx]
        else:
            raise IndexError("Channel index out of range.")

    def get_latest_snapshot(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return the latest buffered (audio, spectrogram) snapshot."""
        if self.snapshot_queue:
            return self.snapshot_queue[-1]
        else:
            return None

    def stop(self) -> None:
        """Stop audio processing and release resources."""
        if hasattr(self, 'stream') and self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
                print("Output stream stopped.")
            except Exception as e:
                print(f"Error stopping output stream: {e}")

        if self.is_streaming and hasattr(self, 'input_stream') and self.input_stream is not None:
            try:
                self.input_stream.stop()
                self.input_stream.close()
                print("Input stream stopped.")
            except Exception as e:
                print(f"Error stopping input stream: {e}")

        # Stop any other resources if needed (timers etc.)
        print("AudioProcessor stopped cleanly.")

    def latest_audio(self, n_samples: int | None = None) -> np.ndarray:
        """
        Copy the last ``n_samples`` mono-mixed samples from the rolling buffer.
        If *n_samples* is None, returns the entire buffer (latest first).
        """
        #   mono mix: mean over channels
        audio = self.audio_buffer.mean(axis=1)
        if n_samples is None or n_samples >= len(audio):
            return audio.copy()
        return audio[-n_samples:].copy()
