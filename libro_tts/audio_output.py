"""Owned sibling output transactions and ordered incremental PCM16 WAV writes."""
from __future__ import annotations

import os
from pathlib import Path
import stat
from uuid import uuid4
import wave

import numpy as np


class AudioOutputError(RuntimeError):
    """A storage/format failure that serial resynthesis cannot repair."""


class AtomicAudioFile:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.temporary = None
        self.handle = None

    def __enter__(self):
        self.destination = self.output_path.resolve()
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.previous_mode = stat.S_IMODE(self.destination.stat().st_mode) if self.destination.is_file() else None
        candidate = self.destination.with_name(f'.libro-audio-{uuid4().hex}.tmp{self.destination.suffix}')
        descriptor = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o666)
        self.temporary = candidate
        try:
            self.handle = os.fdopen(descriptor, 'wb')
        except BaseException:
            os.close(descriptor)
            candidate.unlink(missing_ok=True)
            raise
        return self

    def publish(self) -> Path:
        self.handle.flush()
        if os.fstat(self.handle.fileno()).st_size == 0:
            raise RuntimeError(f"Audio encoder produced an empty output for '{self.output_path}'.")
        if self.previous_mode is not None:
            os.fchmod(self.handle.fileno(), self.previous_mode)
        os.fsync(self.handle.fileno())
        self.handle.close()
        os.replace(self.temporary, self.destination)
        return self.output_path

    def __exit__(self, *_exc):
        try:
            if self.handle is not None:
                self.handle.close()
        finally:
            if self.temporary is not None:
                self.temporary.unlink(missing_ok=True)


class StreamingWavOutput:
    """One text chunk at a time; publish only a complete synchronized WAV."""
    def __init__(self, output_path: Path):
        self.transaction = AtomicAudioFile(output_path)
        self.writer = None
        self.frames = 0
        self.channels = None
        self.sample_rate = None
        self.source_pcm_bytes = 0

    def __enter__(self):
        self.transaction.__enter__()
        return self

    def append(self, audio: np.ndarray, sample_rate: int) -> None:
        try:
            self._append(audio, sample_rate)
        except OSError as exc:
            raise AudioOutputError(f'Failed incremental WAV output at {self.transaction.output_path}: {exc}') from exc

    def _append(self, audio: np.ndarray, sample_rate: int) -> None:
        channels = 1 if audio.ndim == 1 else audio.shape[1]
        if (self.frames + audio.shape[0]) * channels * 2 > 0xFFFFFFFF - 36:
            raise AudioOutputError('WAV exceeds the RIFF 4 GiB limit; split the input or use another audio format.')
        if self.writer is None:
            self.channels = channels
            self.sample_rate = sample_rate
            self.writer = wave.open(self.transaction.handle, 'wb')
            self.writer.setnchannels(channels)
            self.writer.setsampwidth(2)
            self.writer.setframerate(sample_rate)
        elif channels != self.channels or sample_rate != self.sample_rate:
            raise RuntimeError('Inconsistent sample rate or channels in incremental WAV output.')
        # Match the pinned mlx-audio PCM16 conversion, without its full-file
        # Python sample list or combined float32 waveform allocation.
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16) if audio.dtype.kind == 'f' else audio.astype(np.int16, copy=False)
        self.writer.writeframesraw(pcm.astype('<i2', copy=False).tobytes())
        self.frames += audio.shape[0]
        self.source_pcm_bytes += audio.nbytes

    def reset(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.transaction.handle.seek(0)
        self.transaction.handle.truncate()
        self.frames = 0
        self.source_pcm_bytes = 0
        self.channels = self.sample_rate = None

    def publish(self) -> Path:
        if self.frames == 0:
            raise RuntimeError('No audio returned for incremental WAV output.')
        self.writer.close()
        self.writer = None
        return self.transaction.publish()

    def __exit__(self, *exc):
        try:
            if self.writer is not None:
                self.writer.close()
        finally:
            self.transaction.__exit__(*exc)
