#!/usr/bin/env python3
"""Optional heavy integration runner for all supported TTS models.

This script is intentionally opt-in because it downloads and runs large models.
Usage:
  .venv/bin/python tests/integration_run_all_models.py --run
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libro_tts.catalog import list_model_keys
from libro_tts.env import validate_project_virtualenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heavy integration checks across all TTS models.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually execute integration generation runs.",
    )
    parser.add_argument(
        "--offline-second-pass",
        action="store_true",
        help="After first pass, run a second pass with --offline to validate local reuse.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.run:
        print("Skipped. Re-run with --run to execute heavy integration tests.")
        return 0

    validate_project_virtualenv()
    root = Path(__file__).resolve().parent.parent
    fixtures = [
        root / "input.txt",
        root / "tests" / "fixtures" / "public_domain" / "alice_excerpt.txt",
        root / "tests" / "fixtures" / "public_domain" / "federalist_1_excerpt.txt",
    ]

    output_dir = root / "tests_output" / "integration"
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_key in list_model_keys():
        for fixture in fixtures:
            out_prefix = output_dir / f"{model_key}_{fixture.stem}"
            cmd = [
                "bash",
                str(root / "run.sh"),
                str(fixture),
                "--model",
                model_key,
                "--output",
                str(out_prefix),
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

            if args.offline_second_pass:
                offline_prefix = output_dir / f"{model_key}_{fixture.stem}_offline"
                offline_cmd = cmd + ["--offline", "--output", str(offline_prefix)]
                print("Running:", " ".join(offline_cmd))
                subprocess.run(offline_cmd, check=True)

    print("Integration run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
