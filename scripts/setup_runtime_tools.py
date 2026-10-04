"""Expose the locked wheel's FFmpeg binary to mlx-audio without global tools."""

from pathlib import Path
import os
import sys


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    prefix = root / ".venv"
    if Path(sys.prefix).absolute() != prefix or sys.prefix == sys.base_prefix:
        raise SystemExit("Run this setup helper using the project's .venv/bin/python.")

    # Do not allow caller configuration to select an external encoder.
    os.environ.pop("IMAGEIO_FFMPEG_EXE", None)
    import imageio_ffmpeg

    binary = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    if not binary.is_relative_to(prefix.resolve()):
        raise SystemExit(f"The locked FFmpeg wheel did not provide a local binary: {binary}")
    destination = prefix / "bin" / "ffmpeg"
    if destination.exists() or destination.is_symlink():
        if not destination.is_symlink():
            raise SystemExit(f"Refusing to replace a non-symlink encoder: {destination}")
        destination.unlink()
    destination.symlink_to(os.path.relpath(binary, destination.parent))
    print(f"Local FFmpeg: {destination} -> {binary.name}")
    print(f"FFmpeg version: {imageio_ffmpeg.get_ffmpeg_version()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
