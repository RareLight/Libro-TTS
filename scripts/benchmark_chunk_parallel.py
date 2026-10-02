#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libro_tts.logging_config import configure_logging
from libro_tts.env import validate_mlx_backend_preflight
from libro_tts.env import validate_tts_model_runtime_support
from libro_tts.runtime import (
    RuntimeOptions,
    _chunk_text_for_generation,
    _collect_generation_audio,
    _load_runtime_model,
    _populate_missing_ref_text,
    _read_input_text,
    _resolve_model_reference,
    _validate_model_generate_kwargs,
    build_generate_kwargs,
)
from libro_tts.store import ModelStore

_WORKER_MODEL = None
_WORKER_KWARGS: dict[str, Any] | None = None


def _worker_init(model_reference: str, kwargs_template: dict[str, Any]) -> None:
    global _WORKER_MODEL, _WORKER_KWARGS
    _WORKER_MODEL = _load_runtime_model(model_reference, configure_logging(verbose=False))
    _WORKER_KWARGS = kwargs_template


def _worker_noop(_index: int) -> int:
    return 0


def _worker_generate(item: tuple[int, str]) -> tuple[int, float, int, int]:
    if _WORKER_MODEL is None or _WORKER_KWARGS is None:
        raise RuntimeError("Worker model was not initialized.")

    index, text_chunk = item
    kwargs = dict(_WORKER_KWARGS)
    kwargs["text"] = text_chunk
    started = time.perf_counter()
    results = _WORKER_MODEL.generate(**kwargs)
    segments, sample_rate = _collect_generation_audio(
        results=results,
        fallback_sample_rate=getattr(_WORKER_MODEL, "sample_rate", None),
    )
    elapsed = time.perf_counter() - started
    samples = sum(int(segment.size) for segment in segments)
    return index, elapsed, samples, sample_rate


def _prepare_chunks(
    *,
    input_file: Path,
    model_store: ModelStore,
    options: RuntimeOptions,
    logger,
    max_chunks: int | None,
) -> tuple[str, str, dict[str, Any], list[str]]:
    text = _read_input_text(input_file=input_file, options=options, logger=logger)
    model_reference, spec = _resolve_model_reference(model_store=model_store, options=options, logger=logger)
    resolved_model_key = spec.key if spec is not None else options.model_key

    kwargs = build_generate_kwargs(
        text=text,
        spec=spec,
        model_key=options.model_key,
        voice=options.voice,
        speed=options.speed,
        lang_code=options.lang_code,
        max_tokens=options.max_tokens,
        verbose=False,
    )
    _populate_missing_ref_text(
        model_key=resolved_model_key,
        kwargs=kwargs,
        logger=logger,
        verbose=False,
    )
    _validate_model_generate_kwargs(model_key=resolved_model_key, kwargs=kwargs)

    text_chunks = _chunk_text_for_generation(kwargs.pop("text", ""), kwargs.get("speed"))
    if max_chunks is not None and max_chunks > 0:
        text_chunks = text_chunks[:max_chunks]
    if len(text_chunks) < 2:
        raise RuntimeError(
            f"Need at least 2 synthesis chunks for benchmark; got {len(text_chunks)}. "
            "Use a longer input file or lower max chunk duration constraints."
        )
    return model_reference, resolved_model_key, kwargs, text_chunks


def _run_serial(model_reference: str, kwargs_template: dict[str, Any], chunks: list[str], logger) -> dict[str, Any]:
    model = _load_runtime_model(model_reference, logger)
    chunk_times: list[float] = []
    total_samples = 0
    sample_rate = None

    started = time.perf_counter()
    for chunk in chunks:
        kwargs = dict(kwargs_template)
        kwargs["text"] = chunk
        chunk_start = time.perf_counter()
        results = model.generate(**kwargs)
        segments, sr = _collect_generation_audio(
            results=results,
            fallback_sample_rate=getattr(model, "sample_rate", None),
        )
        chunk_times.append(time.perf_counter() - chunk_start)
        total_samples += sum(int(segment.size) for segment in segments)
        sample_rate = sr if sample_rate is None else sample_rate
    elapsed = time.perf_counter() - started

    return {
        "mode": "serial",
        "wall_seconds": elapsed,
        "chunks": len(chunks),
        "chunks_per_second": len(chunks) / elapsed,
        "avg_chunk_seconds": sum(chunk_times) / len(chunk_times),
        "total_samples": total_samples,
        "sample_rate": sample_rate,
    }


def _run_parallel(
    model_reference: str,
    kwargs_template: dict[str, Any],
    chunks: list[str],
    workers: int,
) -> dict[str, Any]:
    indexed_chunks = list(enumerate(chunks))

    with mp.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(model_reference, kwargs_template),
    ) as pool:
        pool.map(_worker_noop, range(workers))
        started = time.perf_counter()
        results = pool.map(_worker_generate, indexed_chunks)
        elapsed = time.perf_counter() - started

    results.sort(key=lambda item: item[0])
    chunk_times = [item[1] for item in results]
    total_samples = sum(item[2] for item in results)
    sample_rates = {item[3] for item in results}
    if len(sample_rates) != 1:
        raise RuntimeError(f"Inconsistent sample rates from parallel workers: {sample_rates}")

    return {
        "mode": f"parallel_{workers}",
        "wall_seconds": elapsed,
        "chunks": len(chunks),
        "chunks_per_second": len(chunks) / elapsed,
        "avg_chunk_seconds": sum(chunk_times) / len(chunk_times),
        "total_samples": total_samples,
        "sample_rate": sample_rates.pop(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark serial vs parallel chunk generation for Libro-TTS."
    )
    parser.add_argument("--input-file", type=Path, required=True, help="Path to long .txt input.")
    parser.add_argument("--model", type=str, default="spark", help="Model key to benchmark.")
    parser.add_argument("--voice", type=str, default="Emma", help="Voice/prompt name.")
    parser.add_argument("--workers", type=int, default=2, help="Parallel worker count.")
    parser.add_argument("--max-chunks", type=int, default=8, help="Limit benchmark chunk count.")
    parser.add_argument("--offline", action="store_true", help="Disable downloads for model resolution.")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip MLX backend preflight (advanced troubleshooting only).",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    logger = configure_logging(verbose=False)
    try:
        validate_mlx_backend_preflight(
            expected_conda_env="tts",
            skip_preflight=args.skip_preflight,
        )
        validate_tts_model_runtime_support(
            args.model,
            expected_conda_env="tts",
        )
    except Exception as exc:
        print(f"Preflight failed: {exc}")
        return 1

    model_store = ModelStore()
    options = RuntimeOptions(
        model_key=args.model,
        model_path_override=None,
        voice=args.voice,
        speed=None,
        lang_code=None,
        max_tokens=None,
        verbose=False,
        offline=args.offline,
        audio_format="wav",
    )

    model_reference, resolved_model_key, kwargs_template, chunks = _prepare_chunks(
        input_file=args.input_file,
        model_store=model_store,
        options=options,
        logger=logger,
        max_chunks=args.max_chunks,
    )

    serial = _run_serial(model_reference, kwargs_template, chunks, logger)
    parallel = _run_parallel(model_reference, kwargs_template, chunks, workers=args.workers)
    speedup = serial["wall_seconds"] / parallel["wall_seconds"]

    summary = {
        "model_key": resolved_model_key,
        "model_reference": model_reference,
        "chunks_benchmarked": len(chunks),
        "serial": serial,
        "parallel": parallel,
        "speedup": speedup,
        "delta_chunks_per_second": parallel["chunks_per_second"] - serial["chunks_per_second"],
    }

    print("Chunk Parallelization Benchmark")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote benchmark JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
