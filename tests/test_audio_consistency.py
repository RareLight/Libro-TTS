import io
import logging
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import wave
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from libro_tts.catalog import get_model_spec
from libro_tts.progress import ChunkProgressBar
from libro_tts.runtime import (
    RuntimeOptions, _collect_generation_audio, _generate_audio_parallel,
    _generate_audio_serial, _generate_chunk_audio, _to_numpy_audio, generate_tts,
)


def result(audio, rate=24000):
    return SimpleNamespace(audio=audio, sample_rate=rate)


class ProgressRecorder:
    def __init__(self):
        self.completed = 0
        self.peak = 0
        self.events = []

    def update(self, step=1):
        self.completed += step
        self.peak = max(self.peak, self.completed)
        self.events.append(("update", self.completed))

    def reset(self):
        self.completed = 0
        self.events.append(("reset", 0))

    def close(self):
        self.events.append(("close", self.completed))


# Importable pure fixtures run inside real spawned workers, without models or
# backend imports. Exercise the application's pool, batching, IPC, and ordering.
def spawned_audio_fixture(task):
    index, text, _reference, _kwargs, _suppress, _cache = task
    if text == "fail":
        raise RuntimeError("fixture worker failure")
    rate = 16000 if text == "rate-change" else 24000
    audio = np.array([index, index + 0.25], dtype=np.float32)
    if text == "stereo":
        audio = np.column_stack((audio, -audio))
    return index, audio, rate


class WaveformValidationTests(unittest.TestCase):
    def test_mono_row_column_and_one_sample_shapes_preserve_samples(self):
        cases = [np.array([0.25]), np.array([[0.25]]), np.array([[0.25, 0.5]]),
                 np.array([[0.25], [0.5]])]
        for audio in cases:
            with self.subTest(shape=audio.shape):
                actual = _to_numpy_audio(audio)
                self.assertEqual(actual.ndim, 1)
                np.testing.assert_array_equal(actual, audio.reshape(-1))

    def test_channel_last_audio_keeps_layout_values_and_dtype(self):
        audio = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        self.assertIs(_to_numpy_audio(audio), audio)
        chunks, rate = _collect_generation_audio([result(audio), result(audio)], None)
        self.assertEqual(rate, 24000)
        np.testing.assert_array_equal(np.concatenate(chunks), np.vstack((audio, audio)))

    def test_array_like_adapters_and_half_precision_are_supported(self):
        audio = np.array([0.25, -0.5], dtype=np.float16)
        for value in (audio, SimpleNamespace(tolist=lambda: audio.tolist()), audio.tolist()):
            with self.subTest(type=type(value)):
                actual = _to_numpy_audio(value)
                np.testing.assert_array_equal(actual, [0.25, -0.5])
                self.assertIn(actual.dtype, (np.dtype("float32"), np.dtype("float64")))

    def test_scalar_high_dimension_non_numeric_complex_and_nonfinite_are_rejected(self):
        cases = (np.array(0.1), np.zeros((1, 2, 3)), np.array(["sample"]),
                 np.array([0.1], dtype=object), np.array([True]), np.array([1 + 2j]),
                 np.array([float("nan")]), np.array([float("inf")]), np.array([-float("inf")]))
        for audio in cases:
            with self.subTest(audio=audio), self.assertRaises(RuntimeError):
                _to_numpy_audio(audio)

    def test_empty_and_metadata_only_results_are_skipped_but_not_counted(self):
        progress = ProgressRecorder()
        chunks, rate = _collect_generation_audio(
            [None, result(None), result(np.array([])), result(np.array([0.1]))],
            24000, progress=progress)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(rate, 24000)
        self.assertEqual(progress.completed, 1)

    def test_empty_generation_is_an_error(self):
        for results in ([], [None], [result(None)], [result(np.array([]))], [result(np.empty((3, 0)))]):
            with self.subTest(results=results), self.assertRaisesRegex(RuntimeError, "without any audio"):
                _collect_generation_audio(results, 24000)

    def test_invalid_rates_from_result_or_model_are_rejected(self):
        for rate in (0, -1, 24000.5, float("nan"), float("inf"), "24000", True, complex(24000)):
            for fallback in (True, False):
                with self.subTest(rate=rate, fallback=fallback), \
                        self.assertRaisesRegex(RuntimeError, "positive integer"):
                    _collect_generation_audio([result(np.array([0.1]), None if fallback else rate)],
                                              rate if fallback else None)

    def test_integral_numeric_rates_and_deferred_rate_are_supported(self):
        for rate in (24000, 24000.0, np.int64(24000), np.float32(24000)):
            with self.subTest(rate=rate):
                _chunks, actual = _collect_generation_audio(
                    [result(np.array([0.1]), None), result(np.array([0.2]), rate)], None)
                self.assertEqual(actual, 24000)
                self.assertIsInstance(actual, int)
        with self.assertRaisesRegex(RuntimeError, "did not expose a sample rate"):
            _collect_generation_audio([result(np.array([0.1]), None)], None)

    def test_result_and_model_rate_must_agree(self):
        for results, fallback in (([result(np.array([0.1]), 16000)], 24000),
                                  ([result(np.array([0.1])), result(np.array([0.2]), 16000)], None)):
            progress = ProgressRecorder()
            with self.subTest(fallback=fallback), self.assertRaisesRegex(RuntimeError, "Inconsistent sample rates"):
                _collect_generation_audio(results, fallback, progress=progress)
            self.assertEqual(progress.completed, 0 if fallback is not None else 1)

    def test_channel_mismatches_fail_before_concatenation_and_progress(self):
        for first, second in ((np.ones(3), np.ones((3, 2))),
                              (np.ones((3, 2)), np.ones((3, 3)))):
            progress = ProgressRecorder()
            with self.subTest(shapes=(first.shape, second.shape)), \
                    self.assertRaisesRegex(RuntimeError, "Inconsistent channel layouts"):
                _collect_generation_audio([result(first), result(second)], 24000, progress=progress)
            self.assertEqual(progress.completed, 1)

    def test_segment_order_and_text_chunk_order_are_preserved(self):
        model = SimpleNamespace(sample_rate=24000, generate=Mock(side_effect=[
            [result(np.array([[0.1]])), result(np.array([0.2, 0.3]))],
            [result(np.array([[0.4], [0.5]]))]]))
        progress = ProgressRecorder()
        chunks, rate = _generate_audio_serial(model=model, kwargs_template={"voice": "fixture"},
            text_chunks=["first", "second"], progress=progress)
        np.testing.assert_array_equal(np.concatenate(chunks), [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(rate, 24000)
        self.assertEqual(progress.completed, 2)
        self.assertEqual([call.kwargs["text"] for call in model.generate.call_args_list], ["first", "second"])

    def test_serial_cross_text_chunk_mismatch_does_not_count_bad_chunk(self):
        for second in (result(np.ones(3), 16000), result(np.ones((3, 2)))):
            model = SimpleNamespace(sample_rate=None, generate=Mock(side_effect=[
                [result(np.ones(3))], [second]]))
            progress = ProgressRecorder()
            with self.subTest(second=second), self.assertRaisesRegex(RuntimeError, "Inconsistent"):
                _generate_audio_serial(model=model, kwargs_template={}, text_chunks=["first", "second"],
                                       progress=progress)
            self.assertEqual(progress.completed, 1)

    def test_one_sample_chunk_generation_remains_a_waveform(self):
        model = SimpleNamespace(sample_rate=24000, generate=lambda **_kwargs: [result(np.array([[0.25]]))])
        audio, rate = _generate_chunk_audio(model=model, kwargs_template={}, text_chunk="hello")
        self.assertEqual(audio.shape, (1,))
        self.assertEqual(rate, 24000)


class ParallelAndFallbackTests(unittest.TestCase):
    def test_real_spawn_pool_preserves_order_across_batches(self):
        progress = ProgressRecorder()
        with patch("libro_tts.runtime._parallel_worker_generate", spawned_audio_fixture):
            chunks, rate = _generate_audio_parallel(model_reference="fixture", kwargs_template={},
                text_chunks=["one", "two", "three", "four", "five"], workers=2, batch_size=2, progress=progress)
        self.assertEqual(rate, 24000)
        np.testing.assert_array_equal(np.concatenate(chunks),
                                      [0, 0.25, 1, 1.25, 2, 2.25, 3, 3.25, 4, 4.25])
        self.assertEqual(progress.completed, 5)

    def test_real_spawn_failure_and_mismatches_keep_partial_progress_bounded(self):
        for text, message in (("fail", "fixture worker failure"),
                              ("rate-change", "Inconsistent sample rates"),
                              ("stereo", "Inconsistent channel layouts")):
            progress = ProgressRecorder()
            with self.subTest(text=text), patch("libro_tts.runtime._parallel_worker_generate", spawned_audio_fixture), \
                    self.assertRaisesRegex(RuntimeError, message):
                _generate_audio_parallel(model_reference="fixture", kwargs_template={},
                    text_chunks=["one", "two", text], workers=2, batch_size=2, progress=progress)
            self.assertEqual(progress.completed, 2)

    def test_parallel_results_are_sorted_before_assembly(self):
        pool = MagicMock()
        pool.__enter__.return_value = pool
        pool.map.return_value = [(1, np.array([2.]), 24000), (0, np.array([1.]), 24000)]
        with patch("libro_tts.runtime.mp.get_context", return_value=SimpleNamespace(Pool=Mock(return_value=pool))):
            chunks, _rate = _generate_audio_parallel(model_reference="fixture", kwargs_template={},
                text_chunks=["first", "second"], workers=2, batch_size=2)
        np.testing.assert_array_equal(np.concatenate(chunks), [1., 2.])

    def run_fallback(self, *, serial_fails=False, real_pool=False, first_results=None, failure_message="non-finite"):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            output = root / "output.wav"
            output.write_bytes(b"previous complete audio")
            progress = ProgressRecorder()
            if first_results is None:
                first_results = [result(np.array([float("nan")]) if serial_fails else np.array([0.1, 0.2]))]
            model = SimpleNamespace(sample_rate=24000, generate=Mock(side_effect=[
                first_results,
                [result(np.array([0.3, 0.4]))], [result(np.array([0.5, 0.6]))]]))
            stack.enter_context(patch("libro_tts.runtime._resolve_model_reference", return_value=(str(root), get_model_spec("soprano"))))
            stack.enter_context(patch("libro_tts.runtime.validate_model_load_preflight"))
            stack.enter_context(patch("libro_tts.runtime.prepare_model_assets"))
            stack.enter_context(patch("libro_tts.runtime._load_runtime_model", return_value=model))
            stack.enter_context(patch("libro_tts.runtime._chunk_text_for_generation", return_value=["one", "two", "fail"]))
            stack.enter_context(patch("libro_tts.runtime.ChunkProgressBar", return_value=progress))
            get_writer = stack.enter_context(patch("libro_tts.runtime._get_audio_write_fn"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            if real_pool:
                stack.enter_context(patch("libro_tts.runtime._parallel_worker_generate", spawned_audio_fixture))
            else:
                pool = MagicMock()
                pool.__enter__.return_value = pool
                pool.map.side_effect = [[(0, np.array([99.]), 24000)], RuntimeError("worker failed")]
                stack.enter_context(patch("libro_tts.runtime.mp.get_context", return_value=SimpleNamespace(Pool=Mock(return_value=pool))))
            options = RuntimeOptions("soprano", None, None, None, None, None, False, True,
                                     parallel_workers=2, parallel_max_chunks=1)
            kwargs = dict(text="Hello", output_prefix=str(output), model_store=SimpleNamespace(root_dir=root),
                          options=options, logger=Mock(spec=logging.Logger))
            if serial_fails:
                with self.assertRaisesRegex(RuntimeError, failure_message):
                    generate_tts(**kwargs)
                self.assertEqual(output.read_bytes(), b"previous complete audio")
            else:
                generate_tts(**kwargs)
                with wave.open(str(output)) as decoded:
                    self.assertEqual((decoded.getframerate(), decoded.getnchannels(), decoded.getnframes()), (24000, 1, 6))
                    samples = np.frombuffer(decoded.readframes(6), dtype='<i2') / 32767.0
                np.testing.assert_allclose(samples, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], atol=1.0 / 32767)
            get_writer.assert_not_called()
            self.assertIn(("reset", 0), progress.events)
            self.assertEqual(progress.events[-1], ("close", 0 if serial_fails else 3))
            self.assertLessEqual(progress.peak, 3)
            self.assertEqual(len(list(root.glob(".libro-audio-*"))), 0)
            return progress

    def test_partial_parallel_failure_restarts_all_text_and_progress_without_duplicate_audio(self):
        progress = self.run_fallback()
        self.assertEqual(progress.events, [("update", 1), ("reset", 0),
                                          ("update", 1), ("update", 2), ("update", 3), ("close", 3)])

    def test_real_worker_failure_runs_full_serial_fallback(self):
        self.run_fallback(real_pool=True)

    def test_failed_serial_retry_closes_progress_and_preserves_previous_output(self):
        self.run_fallback(serial_fails=True)

    def test_invalid_serial_retry_results_never_reach_encoder_or_replace_output(self):
        cases = (([], "without any audio"), ([result(np.array([]))], "without any audio"),
                 ([result(np.array([0.1]), 0)], "positive integer"),
                 ([result(np.ones(3)), result(np.ones((3, 2)))], "Inconsistent channel layouts"),
                 ([result(np.array(["invalid"]))], "real numeric"))
        for results, message in cases:
            with self.subTest(message=message):
                self.run_fallback(serial_fails=True, first_results=results, failure_message=message)


class ProgressResetTests(unittest.TestCase):
    def test_tty_progress_reset_delegates_to_bar_and_retains_total(self):
        with patch("sys.stdout.isatty", return_value=True), patch("tqdm.auto.tqdm") as tqdm:
            progress = ChunkProgressBar("Generating", total=3)
            progress.update()
            progress.reset()
            progress.close()
            tqdm.assert_called_once_with(total=3, desc="Generating", unit="chunk", dynamic_ncols=True, leave=False)
            self.assertEqual(tqdm.return_value.method_calls,
                             [unittest.mock.call.update(1), unittest.mock.call.reset(), unittest.mock.call.close()])

    def test_disabled_and_non_tty_progress_reset_are_noops(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch("sys.stdout.isatty", return_value=False):
                progress = ChunkProgressBar("Generating", enabled=enabled, total=3)
                progress.update()
                progress.reset()
                progress.close()
                self.assertIsNone(progress._bar)
