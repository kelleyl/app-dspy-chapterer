"""Lightweight types used by the chaptering recipe."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TimedItem:
    """A piece of text anchored to a [start_ms, end_ms] interval."""

    start_ms: int
    end_ms: int
    text: str


@dataclass
class Chapter:
    """A predicted chapter with start/end (ms) and a title."""

    start: int
    end: int
    title: str


@dataclass
class VideoSignals:
    """Multi-modal signals extracted from a MMIF for a single video."""

    duration_ms: int
    asr: list[TimedItem] = field(default_factory=list)
    swt_timeframes: list[TimedItem] = field(default_factory=list)
    chyron_ocr: list[TimedItem] = field(default_factory=list)
    visual_captions: list[TimedItem] = field(default_factory=list)
