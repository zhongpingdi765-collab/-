from __future__ import annotations

import unicodedata


def prepare_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\n", "").strip()
