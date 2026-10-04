import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from libro_tts.audio_output import StreamingWavOutput
from libro_tts.runtime import _generate_audio_serial, _to_numpy_audio, _write_output_audio
from types import SimpleNamespace


def read_pcm(path):
    with wave.open(str(path), 'rb') as source:
        metadata = (source.getnchannels(), source.getframerate(), source.getnframes(), source.getsampwidth())
        return metadata, source.readframes(source.getnframes())


class StreamingWavTests(unittest.TestCase):
    def test_streamed_pcm_matches_pinned_writer_for_mono_stereo_and_numeric_types(self):
        cases = (np.array([-2., -0.25, 0.125, 0.5, 2.], dtype=np.float32),
                 np.array([-2., -0.25, 0.125, 0.5, 2.], dtype=np.float64),
                 np.array([0.25], dtype=np.float16),
                 np.array([-32768, -1, 0, 1, 32767], dtype=np.int16),
                 np.array([-40000, 0, 40000], dtype=np.int32),
                 np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for case in cases:
                with self.subTest(dtype=case.dtype, shape=case.shape):
                    audio = _to_numpy_audio(case)
                    reference = _write_output_audio(output_target=str(root / 'reference.wav'), audio_format='wav',
                                                    audio=audio, sample_rate=24000)
                    target = root / 'streamed.wav'
                    previous = target.read_bytes() if target.exists() else None
                    with StreamingWavOutput(target) as output:
                        # Single-frame stereo is already normalized at the input boundary.
                        for chunk in np.array_split(audio, 2) if audio.shape[0] > 1 else [audio]:
                            output.append(chunk, 24000)
                        self.assertEqual(target.read_bytes() if target.exists() else None, previous)
                        output.publish()
                    self.assertEqual(read_pcm(target), read_pcm(reference))
                    self.assertFalse(list(root.glob('.libro-audio-*')))

    def test_full_serial_path_releases_chunks_and_keeps_order(self):
        model = SimpleNamespace(sample_rate=24000, generate=lambda **kwargs: [SimpleNamespace(
            audio=np.array([0.1 if kwargs['text'] == 'first' else 0.2], dtype=np.float32), sample_rate=24000)])
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'output.wav'
            with StreamingWavOutput(target) as output:
                retained, rate = _generate_audio_serial(model=model, kwargs_template={}, text_chunks=['first', 'second'], audio_sink=output)
                self.assertEqual(retained, [])
                self.assertEqual(rate, 24000)
                self.assertEqual(output.frames, 2)
                output.publish()
            metadata, pcm = read_pcm(target)
            self.assertEqual(metadata, (1, 24000, 2, 2))
            np.testing.assert_array_equal(np.frombuffer(pcm, dtype='<i2'), [3276, 6553])

    def test_reset_removes_partial_audio_and_shorter_retry_has_no_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'output.wav'
            with StreamingWavOutput(target) as output:
                output.append(np.ones(1000, dtype=np.float32), 24000)
                output.reset()
                self.assertEqual((output.frames, output.source_pcm_bytes), (0, 0))
                output.append(np.array([0.25], dtype=np.float32), 16000)
                output.publish()
            metadata, pcm = read_pcm(target)
            self.assertEqual(metadata, (1, 16000, 1, 2))
            self.assertEqual(len(pcm), 2)
            self.assertEqual(target.stat().st_size, 46)

    def test_empty_output_and_stream_failure_preserve_previous_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'output.wav'
            target.write_bytes(b'complete previous audio')
            for failure in ('empty', 'generation', 'layout', 'rate', 'riff-limit'):
                with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                    with StreamingWavOutput(target) as output:
                        if failure == 'empty':
                            output.publish()
                        output.append(np.array([0.1, 0.2], dtype=np.float32), 24000)
                        if failure == 'generation':
                            raise RuntimeError('generation failed')
                        if failure == 'layout':
                            output.append(np.ones((2, 2), dtype=np.float32), 24000)
                        if failure == 'rate':
                            output.append(np.ones(2, dtype=np.float32), 16000)
                        if failure == 'riff-limit':
                            output.frames = 2**32
                            output.append(np.ones(1, dtype=np.float32), 24000)
                self.assertEqual(target.read_bytes(), b'complete previous audio')
                self.assertFalse(list(Path(temporary).glob('.libro-audio-*')))

    def test_disk_sync_and_rename_failures_preserve_previous_file(self):
        for call in ('os.fsync', 'os.replace'):
            with self.subTest(call=call), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / 'output.wav'
                target.write_bytes(b'previous')
                with self.assertRaises(OSError), patch('libro_tts.audio_output.' + call, side_effect=OSError('failed')):
                    with StreamingWavOutput(target) as output:
                        output.append(np.ones(3, dtype=np.float32), 24000)
                        output.publish()
                self.assertEqual(target.read_bytes(), b'previous')
                self.assertFalse(list(Path(temporary).glob('.libro-audio-*')))

    def test_chunk_disk_failure_is_classified_as_nonretryable_and_cleans_up(self):
        from libro_tts.audio_output import AudioOutputError
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'output.wav'
            target.write_bytes(b'previous')
            with self.assertRaisesRegex(AudioOutputError, 'disk full'):
                with StreamingWavOutput(target) as output:
                    output.append(np.ones(3, dtype=np.float32), 24000)
                    with patch.object(output.writer, 'writeframesraw', side_effect=OSError('disk full')):
                        output.append(np.ones(3, dtype=np.float32), 24000)
            self.assertEqual(target.read_bytes(), b'previous')
            self.assertFalse(list(Path(temporary).glob('.libro-audio-*')))

    def test_symlink_and_permission_modes_survive_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / 'real.wav'
            target.write_bytes(b'previous')
            target.chmod(0o640)
            alias = root / 'alias.wav'
            alias.symlink_to(target)
            with StreamingWavOutput(alias) as output:
                output.append(np.ones(3, dtype=np.float32), 24000)
                output.publish()
            self.assertTrue(alias.is_symlink())
            self.assertEqual(stat.S_IMODE(os.stat(target).st_mode), 0o640)
            self.assertEqual(read_pcm(alias)[0], (1, 24000, 3, 2))

    def test_parallel_storage_error_does_not_retry_synthesis(self):
        from contextlib import ExitStack
        from unittest.mock import MagicMock
        from libro_tts.audio_output import AudioOutputError
        from libro_tts.catalog import get_model_spec
        from libro_tts.runtime import RuntimeOptions, generate_tts
        import logging
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            target = root / 'output.wav'
            target.write_bytes(b'previous complete output')
            stack.enter_context(patch('libro_tts.runtime._resolve_model_reference', return_value=(str(root), get_model_spec('soprano'))))
            stack.enter_context(patch('libro_tts.runtime.prepare_model_assets'))
            stack.enter_context(patch('libro_tts.runtime.validate_model_load_preflight'))
            stack.enter_context(patch('libro_tts.runtime._chunk_text_for_generation', return_value=['one', 'two', 'three']))
            load = stack.enter_context(patch('libro_tts.runtime._load_runtime_model'))
            pool = MagicMock()
            pool.__enter__.return_value = pool
            pool.map.return_value = [(0, np.array([0.1]), 24000)]
            stack.enter_context(patch('libro_tts.runtime.mp.get_context', return_value=SimpleNamespace(Pool=lambda **_kwargs: pool)))
            stack.enter_context(patch('libro_tts.audio_output.StreamingWavOutput.append', side_effect=AudioOutputError('disk full')))
            with self.assertRaisesRegex(AudioOutputError, 'disk full'):
                generate_tts(text='hello', output_prefix=str(target), model_store=SimpleNamespace(root_dir=root),
                    options=RuntimeOptions('soprano', None, None, None, None, None, True, True), logger=logging.getLogger('test'))
            load.assert_not_called()
            self.assertEqual(target.read_bytes(), b'previous complete output')
            self.assertFalse(list(root.glob('.libro-audio-*')))
