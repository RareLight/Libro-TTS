from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .catalog import (
    DEFAULT_MODEL_KEY,
    canonical_model_type_for_key,
    list_kokoro_english_voices,
    list_model_keys,
    list_model_specs,
    resolve_model_key,
)
from .env import (
    configure_local_cache_environment,
    doctor,
    validate_tts_model_runtime_support,
    validate_mlx_backend_preflight,
    validate_runtime_environment,
)
from .logging_config import configure_logging
from .paths import prepare_runtime_dirs
from .runtime import (
    RuntimeOptions,
    default_output_target_for_input,
    process_batch_dir,
    process_single_file,
)
from .store import ModelStore


def _build_generate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python Libro-tts.py",
        description="Generate audiobook-style TTS audio on Apple Silicon using mlx-audio.",
    )
    parser.add_argument("input", nargs="?", help="Path to a single input text file")
    parser.add_argument(
        "--input-dir",
        "-d",
        type=str,
        help="Input directory for batch processing (.txt files)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path/prefix (single mode) or output directory (batch mode)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=resolve_model_key,
        default=DEFAULT_MODEL_KEY,
        choices=list_model_keys(),
        help="TTS model preset key or alias (e.g., qwen3-tts, sesame, spark-tts).",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help=(
            "Optional voice override. Named-voice models (kokoro/qwen3_tts/voxtral_tts) "
            "accept upstream voice names. Prompt-cloning models (csm/dia/spark/chatterbox) "
            "accept a .wav path or a prompt name from ./reference_voices."
        ),
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Optional speed override. Spark values are normalized to {0.0,0.5,1.0,1.5,2.0}.",
    )
    parser.add_argument(
        "--audio-format",
        type=str,
        default="wav",
        help="Output audio format. WAV is recommended.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Disallow network downloads; require all models to be present locally.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print supported model presets and pinned candidate repos.",
    )
    parser.add_argument(
        "--list-kokoro-voices",
        action="store_true",
        help="List all supported Kokoro English voice names.",
    )
    parser.add_argument(
        "--diag",
        action="store_true",
        help="Print runtime diagnostics and exit.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser


def _print_model_list() -> None:
    print("Supported models:")
    for spec in list_model_specs():
        print()
        print(f"- {spec.key}: {spec.display_name}")
        print(f"  mlx_model_type: {canonical_model_type_for_key(spec.key)}")
        print("  defaults:")
        print(f"    voice: {spec.default_voice if spec.default_voice is not None else 'none'}")
        print(f"    speed: {spec.default_speed if spec.default_speed is not None else 'none'}")
        print(f"    lang : {spec.default_lang_code if spec.default_lang_code is not None else 'none'}")
        print("  repos:")
        for repo in spec.repo_candidates:
            print(f"    - {repo}")


def _print_kokoro_voices() -> None:
    print("Kokoro English voices:")
    for voice in list_kokoro_english_voices():
        print(f"- {voice}")


def _print_diag() -> int:
    report = doctor(include_tts_runtime_probe=True)
    print("Libro-TTS diagnostics")
    print(f"python_executable: {report.python_executable}")
    print(f"python_version: {report.python_version}")
    print(f"conda_env: {report.conda_env}")
    print(f"expected_conda_env: {report.expected_conda_env or 'any'}")
    print("package_versions:")
    for package, version in report.package_versions.items():
        print(f"  - {package}: {version}")
    print("tts_runtime_probe:")
    print(f"  mlx_audio_version: {report.tts_runtime_probe.mlx_audio_version or 'unknown'}")
    if report.tts_runtime_probe.probe_error:
        print("  status: error")
        print(f"  error: {report.tts_runtime_probe.probe_error}")
    else:
        print("  status: ok")
        print(
            "  available_model_types: "
            + (
                ", ".join(report.tts_runtime_probe.available_model_types)
                if report.tts_runtime_probe.available_model_types
                else "none"
            )
        )
        if report.tts_runtime_probe.model_remapping:
            print("  model_remapping:")
            for alias, target in sorted(report.tts_runtime_probe.model_remapping.items()):
                print(f"    - {alias} -> {target}")
    if report.tts_model_support:
        print("tts_model_support:")
        for model_key, status in sorted(report.tts_model_support.items()):
            print(f"  - {model_key}: {status}")
    return 0


def _run_generation(args: argparse.Namespace) -> int:
    if args.list_models:
        _print_model_list()
        return 0

    if args.list_kokoro_voices:
        _print_kokoro_voices()
        return 0

    if args.diag:
        return _print_diag()

    validate_runtime_environment()
    validate_mlx_backend_preflight()
    validate_tts_model_runtime_support(args.model)

    dirs = prepare_runtime_dirs()
    configure_local_cache_environment(dirs["hf_cache_dir"])

    logger = configure_logging(verbose=args.verbose)
    model_store = ModelStore(root_dir=dirs["models_dir"])

    options = RuntimeOptions(
        model_key=args.model,
        model_path_override=None,
        voice=args.voice,
        speed=args.speed,
        lang_code=None,
        max_tokens=None,
        verbose=args.verbose,
        offline=args.offline,
        audio_format=args.audio_format,
        input_encoding="utf-8",
        input_encoding_fallbacks=(),
    )

    if args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output) if args.output else input_dir
        result = process_batch_dir(
            input_dir=input_dir,
            output_dir=output_dir,
            model_store=model_store,
            options=options,
            logger=logger,
        )
        if result.total_files == 0:
            if args.verbose:
                logger.warning("No .txt files found in input directory: %s", input_dir)
            else:
                print(f"No .txt files found in input directory: {input_dir}")
            return 0

        if result.files_failed > 0:
            if args.verbose:
                logger.error(
                    "Batch completed with failures. Files processed: %s/%s. Failed: %s",
                    result.files_processed,
                    result.total_files,
                    result.files_failed,
                )
            else:
                print(
                    "Batch completed with failures. "
                    f"Files processed: {result.files_processed}/{result.total_files}. "
                    f"Failed: {result.files_failed}"
                )
            return 1

        if args.verbose:
            logger.info(
                "Batch processing complete. Files processed: %s/%s",
                result.files_processed,
                result.total_files,
            )
        else:
            print(f"Batch complete. Files processed: {result.files_processed}")
        return 0

    if args.input:
        input_file = Path(args.input)
        output_prefix = (
            args.output
            if args.output
            else default_output_target_for_input(
                input_file=input_file,
                audio_format=args.audio_format,
            )
        )
        process_single_file(
            input_file=input_file,
            output_prefix=output_prefix,
            model_store=model_store,
            options=options,
            logger=logger,
        )
        if args.verbose:
            logger.info("Audio generation complete")
        return 0

    raise SystemExit(
        "You must specify either a single input file or --input-dir, "
        "or use --list-models/--list-kokoro-voices/--diag."
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    try:
        parser = _build_generate_parser()
        args = parser.parse_args(argv)
        return _run_generation(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
