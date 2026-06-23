"""DSPy chaptering recipe: signature, module, and postprocessing.

Single-shot LLM call produces an ordered list of chapters from the video
signals. Post-processing keeps the output well-formed:
  (1) clamp endpoints into [0, duration_ms]
  (2) sort by start
  (3) drop entries with empty title or end <= start
  (4) enforce non-overlapping intervals
  (5) snap boundaries to the nearest TransNet shot inside `snap_window_ms`
  (6) enforce a minimum chapter duration
"""

from __future__ import annotations

from typing import Iterable

import dspy
from pydantic import BaseModel, Field

from .formatting import format_timed_items_for_prompt
from .prompts import OPTIMIZED_INSTRUCTION
from .snap import snap_boundaries_to_shots
from .types import Chapter, VideoSignals


DEFAULT_MIN_CHAPTER_DURATION_MS = 5_000
DEFAULT_SNAP_TO_SHOT_WINDOW_MS = 3_000


class ChapterEntry(BaseModel):
    """One generated chapter."""

    start_ms: int = Field(description="timestamp in ms where the chapter begins")
    end_ms: int = Field(description="timestamp in ms where the chapter ends")
    title: str = Field(description="short descriptive title for the chapter")


class ChapterGeneration(dspy.Signature):
    __doc__ = OPTIMIZED_INSTRUCTION

    duration_ms: int = dspy.InputField(desc="total video duration in ms")
    asr_text: str = dspy.InputField(
        desc="full ASR transcript with timestamps inline (one sentence per line)"
    )
    visual_text: str = dspy.InputField(
        desc="visual signals (chyron OCR, scene labels, captions, shot-change markers) with timestamps; may be empty"
    )
    chapters: list[ChapterEntry] = dspy.OutputField(
        desc="ordered list of chapters, each with start_ms, end_ms, and a short descriptive title"
    )


class M5GenerateRecipe(dspy.Module):
    """Single-shot chapter detection from scratch."""

    def __init__(
        self,
        min_chapter_duration_ms: int = DEFAULT_MIN_CHAPTER_DURATION_MS,
        snap_to_shot_window_ms: int = DEFAULT_SNAP_TO_SHOT_WINDOW_MS,
    ):
        super().__init__()
        self.generate = dspy.Predict(ChapterGeneration)
        self.min_chapter_duration_ms = max(0, int(min_chapter_duration_ms))
        self.snap_to_shot_window_ms = max(0, int(snap_to_shot_window_ms))

    def forward(self, signals: VideoSignals) -> list[Chapter]:
        if signals.duration_ms <= 0:
            return []

        asr_text = format_timed_items_for_prompt(signals.asr)
        visual_text = format_timed_items_for_prompt(
            list(signals.chyron_ocr)
            + list(signals.visual_captions)
            + list(signals.swt_timeframes)
        )

        result = self.generate(
            duration_ms=signals.duration_ms,
            asr_text=asr_text,
            visual_text=visual_text,
        )

        # Each TransNet shot-boundary TimeFrame is a shot-change candidate.
        shot_change_times = [it.start_ms for it in signals.swt_timeframes]

        return self._postprocess(
            getattr(result, "chapters", []) or [],
            duration_ms=signals.duration_ms,
            shot_change_times_ms=shot_change_times,
        )

    def _postprocess(
        self,
        entries: Iterable,
        duration_ms: int,
        shot_change_times_ms: list[int] | None = None,
    ) -> list[Chapter]:
        cleaned: list[Chapter] = []
        for entry in entries:
            try:
                start = int(entry.start_ms) if hasattr(entry, "start_ms") else int(entry["start_ms"])
                end = int(entry.end_ms) if hasattr(entry, "end_ms") else int(entry["end_ms"])
                title = str(entry.title) if hasattr(entry, "title") else str(entry["title"])
            except (TypeError, ValueError, KeyError, AttributeError):
                continue
            start = max(0, min(duration_ms, start))
            end = max(0, min(duration_ms, end))
            if not title:
                continue
            cleaned.append(Chapter(start=start, end=end, title=title))

        if not cleaned:
            return []

        cleaned.sort(key=lambda c: (c.start, c.end))

        for i in range(len(cleaned) - 1):
            if cleaned[i].end > cleaned[i + 1].start:
                cleaned[i] = Chapter(
                    start=cleaned[i].start,
                    end=cleaned[i + 1].start,
                    title=cleaned[i].title,
                )

        if self.snap_to_shot_window_ms > 0 and shot_change_times_ms:
            starts = [c.start for c in cleaned]
            snapped = snap_boundaries_to_shots(
                starts,
                shot_change_times_ms,
                tolerance_ms=self.snap_to_shot_window_ms,
            )
            cleaned = [
                Chapter(start=s, end=c.end, title=c.title)
                for s, c in zip(snapped, cleaned)
            ]
            for i in range(len(cleaned) - 1):
                if cleaned[i].end < cleaned[i].start:
                    cleaned[i] = Chapter(cleaned[i].start, cleaned[i].start, cleaned[i].title)
                if cleaned[i].end > cleaned[i + 1].start:
                    cleaned[i] = Chapter(cleaned[i].start, cleaned[i + 1].start, cleaned[i].title)

        if self.min_chapter_duration_ms > 0:
            for i in range(len(cleaned)):
                desired_end = cleaned[i].start + self.min_chapter_duration_ms
                upper = cleaned[i + 1].start if i + 1 < len(cleaned) else duration_ms
                upper = max(cleaned[i].start, upper)
                if cleaned[i].end < desired_end:
                    cleaned[i] = Chapter(
                        start=cleaned[i].start,
                        end=min(desired_end, upper),
                        title=cleaned[i].title,
                    )
                if cleaned[i].end < cleaned[i].start:
                    cleaned[i] = Chapter(cleaned[i].start, cleaned[i].start, cleaned[i].title)

        return cleaned
