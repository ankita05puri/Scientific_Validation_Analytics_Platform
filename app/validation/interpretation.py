"""Reusable interpretation helpers for validation reports."""

from __future__ import annotations


def build_sectioned_interpretation(sections: dict[str, str]) -> str:
    """Build a plain-text sectioned interpretation for UI and reports."""

    return "\n\n".join(f"{heading}: {text}" for heading, text in sections.items())

