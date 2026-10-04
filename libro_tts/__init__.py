"""Libro-TTS modular runtime package."""

from .bootstrap import configure_local_cache_environment

# Package imports (including multiprocessing workers) establish cache roots
# before any module can import Hugging Face or the synthesis stack.
configure_local_cache_environment()


def main(argv: list[str] | None = None) -> int:
    from .cli import main as cli_main

    return cli_main(argv)

__all__ = ["main"]
