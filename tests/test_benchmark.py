import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import benchmark_chunk_parallel as benchmark


def slow_benchmark_child(_config, _connection):
    os.setsid()
    time.sleep(5)


class BenchmarkTests(unittest.TestCase):
    def test_empty_text_is_rejected_before_model_acquisition(self):
        import logging
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'empty.txt'
            source.write_text('\ufeff  \n')
            config = {'input_file': str(source), 'models_dir': str(root / 'models'),
                      'model': 'soprano', 'voice': None, 'offline': False}
            with patch('libro_tts.runtime._resolve_model_reference') as resolve:
                with self.assertRaisesRegex(ValueError, 'no text'):
                    benchmark._prepare_chunks(config, logging.getLogger('test'))
                resolve.assert_not_called()

    def test_fixture_profiles_production_pool_and_both_wav_paths_without_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for buffered in (False, True):
                with self.subTest(buffered=buffered):
                    output = root / f'{buffered}.json'
                    command = [sys.executable, '-B', 'scripts/benchmark_chunk_parallel.py', '--fixture',
                               '--offline', '--max-chunks', '3', '--fixture-frames', '16', '--repeats', '1',
                               '--models-dir', str(root / 'models'), '--json-out', str(output)]
                    if buffered:
                        command.append('--buffered-wav')
                    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
                    self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
                    data = json.loads(output.read_text())
                    self.assertEqual(data['fixture'], True)
                    self.assertEqual([mode['workers'] for mode in data['modes']], [1, 2])
                    for mode in data['modes']:
                        run = mode['runs'][0]
                        self.assertTrue(run['success'])
                        self.assertEqual((run['frames'], run['channels'], run['sample_rate']), (48, 1, 24000))
                        self.assertEqual(run['chunk_pcm_bytes'], 192)
                        self.assertEqual(run['combined_pcm_bytes'], 192 if buffered else 0)
                        self.assertEqual(run['audio_seconds'], 48 / 24000)
                        self.assertEqual(run['rtf'], run['wall_seconds'] / run['audio_seconds'])
                    self.assertFalse((root / 'models').exists())

    def test_timeout_stops_owned_benchmark_child(self):
        config = {'timeout': 0.2, 'workers': 2}
        with patch('scripts.benchmark_chunk_parallel._benchmark_child', slow_benchmark_child):
            started = time.perf_counter()
            report = benchmark._run_mode(config)
        self.assertLess(time.perf_counter() - started, 5)
        self.assertFalse(report['runs'][0]['success'])
        self.assertIn('timed out', report['runs'][0]['error'])

    def test_process_group_stop_does_not_signal_unrelated_parent_group(self):
        from types import SimpleNamespace
        terminate = unittest.mock.Mock()
        process = SimpleNamespace(pid=123, terminate=terminate)
        with patch('scripts.benchmark_chunk_parallel.os.getpgid', return_value=456), \
                patch('scripts.benchmark_chunk_parallel.os.killpg') as kill:
            benchmark._stop_process_group(process)
        terminate.assert_called_once()
        kill.assert_not_called()
