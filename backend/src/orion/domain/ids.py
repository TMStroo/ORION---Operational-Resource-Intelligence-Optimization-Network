"""Identifier helpers.

ORION uses plain strings as identifiers everywhere. Identifiers are
namespaced by entity kind in generated scenarios (``T-017``, ``V-004``) so a
reader can tell what a reference points at while reading a JSON export.
"""

from __future__ import annotations

import re

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")

DEPOT_ID = "DEPOT"


def is_valid_id(value: object) -> bool:
    """Return True when *value* is a syntactically valid ORION identifier."""
    return isinstance(value, str) and bool(_ID_RE.match(value))


def validate_id(value: object, *, field: str = "id") -> str:
    """Return *value* as a validated identifier or raise ``ValueError``."""
    if not is_valid_id(value):
        raise ValueError(
            f"{field}={value!r} is not a valid ORION id "
            "(1-64 chars, alphanumeric start, then [A-Za-z0-9_.:-])"
        )
    assert isinstance(value, str)
    return value


def format_id(prefix: str, index: int, *, width: int = 3) -> str:
    """Format a zero-padded identifier such as ``T-017``."""
    return f"{prefix}-{index:0{width}d}"


__all__ = ["DEPOT_ID", "is_valid_id", "validate_id", "format_id"]
