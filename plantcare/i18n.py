"""Tiny translation layer. Add locales/<lang>.json to support a new language."""
import json
from functools import lru_cache

from .config import LOCALE_DIR


@lru_cache(maxsize=None)
def _load(lang: str) -> dict:
    path = LOCALE_DIR / f"{lang}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def t(key: str, lang: str = "en", **fmt) -> str:
    """Look up a message; falls back to English, then to the key itself."""
    text = _load(lang).get(key) or _load("en").get(key) or key
    return text.format(**fmt) if fmt else text


def available_languages() -> list:
    return sorted(p.stem for p in LOCALE_DIR.glob("*.json"))
