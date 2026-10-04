#!/usr/bin/env python3
"""Opt-in production-path timing and sampled RSS, with a model-free fixture mode."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libro_tts.bootstrap import configure_local_cache_environment
from libro_tts.catalog import resolve_model_key
from libro_tts.env import (
    validate_project_virtualenv, validate_runtime_environment,
    validate_mlx_backend_preflight, validate_tts_model_runtime_support,
)
from libro_tts.logging_config import configure_logging
from libro_tts.validation import positive_int, validate_input_path

_NETWORK_BLOCKED = False


def _block_network():
    """Block internet sockets while leaving local multiprocessing IPC usable."""
    global _NETWORK_BLOCKED
    if _NETWORK_BLOCKED:
        return
    import socket
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex
    def guarded(original):
        def call(sock, address):
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                raise RuntimeError('Benchmark offline check blocked an internet socket.')
            return original(sock, address)
        return call
    socket.socket.connect = guarded(connect)
    socket.socket.connect_ex = guarded(connect_ex)
    _NETWORK_BLOCKED = True


def _offline_worker(task):
    _block_network()
    from libro_tts.runtime import _parallel_worker_generate
    return _parallel_worker_generate(task)


def _prepare_chunks(config: dict[str, Any], logger):
    from libro_tts.assets import prepare_model_assets
    from libro_tts.runtime import (
        RuntimeOptions, _read_input_text, _resolve_model_reference, build_generate_kwargs,
        _populate_missing_ref_text, _validate_model_generate_kwargs,
        _chunk_text_for_generation, _resolve_local_kokoro_voice, _validate_voxtral_voice,
    )
    from libro_tts.store import ModelStore
    from libro_tts.text import sanitize_text_for_tts

    options = RuntimeOptions(config['model'], None, config['voice'], None, None, None, False, config['offline'])
    store = ModelStore(Path(config['models_dir']))
    text = _read_input_text(input_file=Path(config['input_file']), options=options, logger=logger)
    text = sanitize_text_for_tts(text.removeprefix('\ufeff'))
    if not text.strip():
        raise ValueError('Input contains no text for synthesis.')
    reference, spec = _resolve_model_reference(model_store=store, options=options, logger=logger)
    key = spec.key if spec else options.model_key
    kwargs = build_generate_kwargs(text=text, spec=spec, model_key=key, voice=options.voice, speed=None,
                                   lang_code=None, max_tokens=None, verbose=False)
    if not kwargs['text'].strip():
        raise ValueError('Input contains no text for synthesis.')
    prepare_model_assets(store, key, reference, options.offline)
    if key == 'kokoro':
        kwargs['voice'] = _resolve_local_kokoro_voice(reference, kwargs['voice'], options.offline, store)
    if key == 'voxtral_tts':
        _validate_voxtral_voice(reference, kwargs['voice'])
    _populate_missing_ref_text(model_key=key, kwargs=kwargs, logger=logger, verbose=False,
                               model_store=store, offline=options.offline)
    _validate_model_generate_kwargs(model_key=key, kwargs=kwargs)
    chunks = _chunk_text_for_generation(kwargs.pop('text'), kwargs.get('speed'))[:config['max_chunks']]
    if len(chunks) < 2:
        raise RuntimeError(f'Need at least two synthesis chunks; got {len(chunks)}.')
    return reference, key, kwargs, chunks


class FixtureModel:
    sample_rate = 24000

    def __init__(self, frames: int):
        self.frames = frames

    def generate(self, **_kwargs):
        import numpy as np
        yield SimpleNamespace(audio=np.full(self.frames, 0.125, dtype=np.float32), sample_rate=self.sample_rate)


def _fixture_worker(task):
    from libro_tts.runtime import _generate_chunk_audio
    index, text, _reference, kwargs, suppress, _cache = task
    audio, rate = _generate_chunk_audio(model=FixtureModel(kwargs['fixture_frames']),
        kwargs_template={}, text_chunk=text, suppress_model_output=suppress)
    return index, audio, rate


def _benchmark_child(config, connection):
    os.setsid()
    logger = configure_logging(verbose=False)
    configure_local_cache_environment(Path(config['models_dir']) / '.hf', offline=config['offline'])
    if config['offline']:
        _block_network()
    reports = []
    for repeat in range(config['repeats']):
        started = time.perf_counter()
        try:
            import numpy as np
            import libro_tts.runtime as runtime
            from libro_tts.assets import local_asset_loading
            if config['fixture']:
                reference, key = 'synthetic-fixture', 'fixture'
                kwargs = {'fixture_frames': config['fixture_frames']}
                chunks = [f'Fixture chunk {index}.' for index in range(config['max_chunks'])]
                runtime._parallel_worker_generate = _fixture_worker
            else:
                reference, key, kwargs, chunks = _prepare_chunks(config, logger)
                if config['offline']:
                    runtime._parallel_worker_generate = _offline_worker
                if config['workers'] > 1 and not runtime._model_supports_parallel_chunk_generation(key):
                    raise RuntimeError(f'{key} has a production parallel reliability guard; use another model.')
            prepared = time.perf_counter()
            sink = runtime.StreamingWavOutput(Path(config['output'])) if not config['buffered_wav'] else None
            with local_asset_loading(Path(config['models_dir']) / '.hf'), (sink if sink is not None else nullcontext()):
                if config['workers'] == 1:
                    model = FixtureModel(config['fixture_frames']) if config['fixture'] else runtime._load_runtime_model(reference, logger)
                loaded = time.perf_counter()
                if config['workers'] == 1:
                    audio_chunks, rate = runtime._generate_audio_serial(model=model, kwargs_template=kwargs,
                        text_chunks=chunks, suppress_model_output=True, audio_sink=sink)
                else:
                    audio_chunks, rate = runtime._generate_audio_parallel(model_reference=reference,
                        kwargs_template=kwargs, text_chunks=chunks, workers=config['workers'],
                        batch_size=config['batch_size'], suppress_model_output=True, audio_sink=sink)
                generated = time.perf_counter()
                if sink is not None:
                    frames, channels = sink.frames, sink.channels
                    pcm_bytes, combined_bytes = sink.source_pcm_bytes, 0
                    concatenated = time.perf_counter()
                    output = sink.publish()
                else:
                    audio = audio_chunks[0] if len(audio_chunks) == 1 else np.concatenate(audio_chunks, axis=0)
                    frames = sum(chunk.shape[0] for chunk in audio_chunks)
                    channels = 1 if audio.ndim == 1 else audio.shape[1]
                    pcm_bytes, combined_bytes = sum(chunk.nbytes for chunk in audio_chunks), audio.nbytes
                    concatenated = time.perf_counter()
                    output = runtime._write_output_audio(output_target=config['output'], audio_format='wav', audio=audio, sample_rate=rate)
                    del audio
            finished = time.perf_counter()
            duration = frames / rate
            reports.append({
                'repeat': repeat + 1, 'cache_state': 'first' if repeat == 0 else 'repeat', 'success': True,
                'chunks': len(chunks), 'frames': frames, 'channels': channels,
                'sample_rate': rate, 'audio_seconds': duration,
                'wall_seconds': finished - started, 'rtf': (finished - started) / duration,
                'prepare_seconds': prepared - started, 'parent_load_seconds': loaded - prepared,
                'generation_seconds': generated - loaded, 'concatenate_seconds': concatenated - generated,
                'encode_seconds': finished - concatenated,
                'chunk_pcm_bytes': pcm_bytes,
                'combined_pcm_bytes': combined_bytes, 'encoded_bytes': output.stat().st_size,
                'parent_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
            })
            # Keep warm model state, but release PCM before the next measurement.
            del audio_chunks
        except Exception as exc:
            reports.append({'repeat': repeat + 1, 'success': False, 'wall_seconds': time.perf_counter() - started,
                            'error': f'{type(exc).__name__}: {exc}'[-4096:]})
            break
    connection.send(reports)
    connection.close()


def _sample_descendant_rss(pid: int) -> int:
    result = subprocess.run(['ps', '-axo', 'pid=,ppid=,rss='], capture_output=True, text=True, check=True)
    rows = [tuple(map(int, line.split())) for line in result.stdout.splitlines() if len(line.split()) == 3]
    descendants = {pid}
    while True:
        expanded = descendants | {child for child, parent, _rss in rows if parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    return sum(rss * 1024 for child, _parent, rss in rows if child in descendants)


def _run_mode(config):
    context = mp.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_benchmark_child, args=(config, sender))
    started = time.perf_counter()
    process.start()
    sender.close()
    peak = 0
    samples = 0
    reports = None
    try:
        while process.is_alive():
            try:
                peak = max(peak, _sample_descendant_rss(process.pid))
                samples += 1
            except (OSError, subprocess.SubprocessError):
                pass
            if receiver.poll(0.2):
                reports = receiver.recv()
                break
            if time.perf_counter() - started > config['timeout']:
                _stop_process_group(process)
                break
        process.join(timeout=5)
        if process.is_alive():
            _stop_process_group(process, signal.SIGKILL)
            process.join()
        if reports is None and receiver.poll():
            reports = receiver.recv()
    except EOFError:
        reports = None
    finally:
        if process.is_alive():
            _stop_process_group(process)
        process.join(timeout=5)
        receiver.close()
    return {'workers': config['workers'], 'runs': reports or [{'success': False, 'error': f'Child exited {process.exitcode} or timed out.'}],
            'sampled_peak_tree_rss_bytes': peak or None, 'rss_samples': samples,
            'process_wall_seconds': time.perf_counter() - started}


def _stop_process_group(process, signum=signal.SIGTERM):
    try:
        if os.getpgid(process.pid) == process.pid:
            os.killpg(process.pid, signum)
        else:
            process.terminate()
    except ProcessLookupError:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file', type=Path, help='Long UTF-8 text; required without --fixture.')
    parser.add_argument('--model', type=resolve_model_key, default='spark')
    parser.add_argument('--voice', default=None, help='Omit to use the catalog default.')
    parser.add_argument('--workers', type=positive_int, default=2)
    parser.add_argument('--parallel-batch-size', type=positive_int, default=12)
    parser.add_argument('--max-chunks', type=positive_int, default=8)
    parser.add_argument('--repeats', type=positive_int, default=2)
    parser.add_argument('--timeout', type=positive_int, default=600, help='Maximum seconds per execution mode.')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--skip-preflight', action='store_true', help='Advanced backend troubleshooting only.')
    parser.add_argument('--models-dir', type=Path, default=PROJECT_ROOT / 'models')
    parser.add_argument('--fixture', action='store_true', help='Use deterministic PCM fixtures; no model/Metal/network.')
    parser.add_argument('--fixture-frames', type=positive_int, default=240000)
    parser.add_argument('--buffered-wav', action='store_true', help='Compare the retained legacy WAV encoder path.')
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args(argv)
    configure_local_cache_environment(args.models_dir / '.hf', offline=args.offline)
    preflight_started = time.perf_counter()
    try:
        validate_project_virtualenv()
        if not args.fixture:
            validate_runtime_environment()
            if args.input_file is None:
                raise ValueError('--input-file is required without --fixture.')
            validate_input_path(args.input_file)
            probe = validate_mlx_backend_preflight(skip_preflight=args.skip_preflight)
            validate_tts_model_runtime_support(args.model, probe=probe)
        if args.max_chunks < 2:
            raise ValueError('--max-chunks must be at least 2.')
    except Exception as exc:
        print(f'Preflight failed: {exc}')
        return 1
    preflight_seconds = time.perf_counter() - preflight_started
    with tempfile.TemporaryDirectory(prefix='libro-benchmark-') as temporary:
        config = {'input_file': str(args.input_file) if args.input_file else None,
                  'model': args.model, 'voice': args.voice, 'offline': args.offline,
                  'models_dir': str(args.models_dir.resolve()), 'fixture': args.fixture,
                  'fixture_frames': args.fixture_frames, 'buffered_wav': args.buffered_wav, 'max_chunks': args.max_chunks,
                  'repeats': args.repeats, 'batch_size': args.parallel_batch_size, 'timeout': args.timeout}
        modes = []
        for workers in (1, args.workers):
            modes.append(_run_mode({**config, 'workers': workers, 'output': str(Path(temporary) / f'{workers}.wav')}))
    summary = {'fixture': args.fixture, 'buffered_wav': args.buffered_wav, 'model_key': args.model if not args.fixture else 'fixture',
               'preflight_seconds': preflight_seconds, 'parallel_batch_size': args.parallel_batch_size,
               'modes': modes,
               'measurement_notes': 'Generation includes new spawn pool startup/model load and full PCM IPC. Runtime timings include WAV encoding (incremental writes are part of generation time); controller/preflight are separate. Repeat serial uses warm model cache; parallel creates a new pool each time. RSS sums sampled child/descendants, may double-count shared pages, and can miss peaks. Fixtures do not predict TTS speed.'}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return 0 if all(run['success'] for mode in modes for run in mode['runs']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
