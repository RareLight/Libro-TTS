from __future__ import annotations

import re

SANITIZE_TEXT_FOR_TTS = True


FOOTNOTE_PATTERN = re.compile(r"\[\s*\d+\s*\]")
WHITESPACE_PATTERN = re.compile(r"[ \t]+")
EXCESS_NEWLINES_PATTERN = re.compile(r"\n{3,}")


def sanitize_text_for_tts(text: str) -> str:
    if not SANITIZE_TEXT_FOR_TTS or not text or not text.strip():
        return text

    out = FOOTNOTE_PATTERN.sub("", text)
    out = WHITESPACE_PATTERN.sub(" ", out)
    out = EXCESS_NEWLINES_PATTERN.sub("\n\n", out)
    return out.strip() if out != text else out
