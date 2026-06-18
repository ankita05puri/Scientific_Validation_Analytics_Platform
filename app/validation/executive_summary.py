"""Reusable executive summary structures for validation modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SummaryCard:
    """Simple label/value card definition used by validation dashboards."""

    label: str
    value: str
    status: str | None = None

