"""DSPy chaptering recipes.

Two recipes:
  * `M5GenerateRecipe`: single-shot LLM call that produces full
    `(start, end, title)` chapters in one structured output.
  * `M5BoundariesRecipe`: two-pass — first call produces just an ordered
    list of boundary timestamps, then a second per-chapter call generates
    titles from the ASR/visual content within each range. Smaller per-call
    outputs (one int per chapter for boundary detection, one string for
    titling), and the boundary task is decoupled from the title task so
    they can be evaluated and optimized independently.

Post-processing for both:
  (1) clamp endpoints into [0, duration_ms]
  (2) sort
  (3) enforce minimum spacing / non-overlap
  (4) snap to nearest TransNet shot inside `snap_window_ms`
  (5) enforce a minimum chapter duration
"""

from __future__ import annotations

from typing import Iterable

import dspy
from pydantic import BaseModel, Field

from .formatting import (
    compact_captions,
    format_events_interleaved,
    format_timed_items_for_prompt,
)
from .prompts import (
    BOUNDARY_INSTRUCTION,
    INTERLEAVED_BOUNDARY_INSTRUCTION,
    OPTIMIZED_INSTRUCTION,
    TITLING_INSTRUCTION,
)
from .snap import snap_boundaries_to_shots
from .types import Chapter, TimedItem, VideoSignals


DEFAULT_MIN_CHAPTER_DURATION_MS = 5_000
DEFAULT_SNAP_TO_SHOT_WINDOW_MS = 3_000


def _items_in_range(items: Iterable[TimedItem], start_ms: int, end_ms: int) -> list[TimedItem]:
    """Items that overlap [start_ms, end_ms]."""
    return [it for it in items if it.start_ms < end_ms and it.end_ms > start_ms]


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
        captions = compact_captions(list(signals.chyron_ocr) + list(signals.visual_captions))
        # `swt_timeframes` only carries "shot" labels — dropped here as
        # raw shot-change rate is noise without semantic content; the
        # snap-to-shot post-processing still uses these times.
        visual_text = format_timed_items_for_prompt(captions)

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


# ----- Boundaries-only recipe (decoupled boundary detection + titling) -----


class BoundaryDetection(dspy.Signature):
    __doc__ = BOUNDARY_INSTRUCTION

    duration_ms: int = dspy.InputField(desc="total video duration in ms")
    asr_text: str = dspy.InputField(
        desc="full ASR transcript with timestamps inline (one sentence per line)"
    )
    visual_text: str = dspy.InputField(
        desc="visual signals (chyron OCR, scene labels, captions, shot-change markers) with timestamps; may be empty"
    )
    boundaries: list[int] = dspy.OutputField(
        desc="ordered ascending list of chapter start times in ms; do not include 0 or duration_ms"
    )


class ChapterTitling(dspy.Signature):
    __doc__ = TITLING_INSTRUCTION

    chapter_start_ms: int = dspy.InputField(desc="chapter start in ms")
    chapter_end_ms: int = dspy.InputField(desc="chapter end in ms")
    asr_text: str = dspy.InputField(
        desc="ASR sentences within this chapter (may be empty)"
    )
    visual_text: str = dspy.InputField(
        desc="visual signals within this chapter (may be empty)"
    )
    title: str = dspy.OutputField(desc="short descriptive title for the chapter")


class M5BoundariesRecipe(dspy.Module):
    """Two-pass chapter detection:
      1. Predict an ordered list of boundary timestamps.
      2. For each [b_i, b_{i+1}] range, predict a title from the ASR /
         visual content inside.

    Args:
        min_chapter_duration_ms: floor on each chapter's duration in ms.
            Boundaries that would create a shorter chapter are merged away.
        snap_to_shot_window_ms: tolerance for snapping each boundary to
            the nearest TransNet shot change. 0 disables snapping.
        skip_titling: if True, skip the second pass and return chapters
            with empty titles. Useful for boundary-only evaluation.
    """

    def __init__(
        self,
        min_chapter_duration_ms: int = DEFAULT_MIN_CHAPTER_DURATION_MS,
        snap_to_shot_window_ms: int = DEFAULT_SNAP_TO_SHOT_WINDOW_MS,
        skip_titling: bool = False,
    ):
        super().__init__()
        self.detect_boundaries = dspy.Predict(BoundaryDetection)
        self.title_chapter = dspy.Predict(ChapterTitling)
        self.min_chapter_duration_ms = max(0, int(min_chapter_duration_ms))
        self.snap_to_shot_window_ms = max(0, int(snap_to_shot_window_ms))
        self.skip_titling = bool(skip_titling)

    def forward(self, signals: VideoSignals) -> list[Chapter]:
        if signals.duration_ms <= 0:
            return []

        asr_text = format_timed_items_for_prompt(signals.asr)
        captions = compact_captions(list(signals.chyron_ocr) + list(signals.visual_captions))
        # Drop raw "shot" labels — see M5GenerateRecipe.forward for the
        # rationale. swt_timeframes is still used for snap-to-shot.
        visual_text = format_timed_items_for_prompt(captions)

        result = self.detect_boundaries(
            duration_ms=signals.duration_ms,
            asr_text=asr_text,
            visual_text=visual_text,
        )

        raw = getattr(result, "boundaries", []) or []
        boundaries = self._clean_boundaries(raw, signals.duration_ms)

        shot_change_times = [it.start_ms for it in signals.swt_timeframes]
        if self.snap_to_shot_window_ms > 0 and shot_change_times and boundaries:
            snapped = snap_boundaries_to_shots(
                boundaries, shot_change_times, tolerance_ms=self.snap_to_shot_window_ms
            )
            boundaries = self._clean_boundaries(snapped, signals.duration_ms)

        # Build (start, end) pairs. Implicit first chapter starts at 0,
        # implicit last ends at duration_ms.
        starts = [0] + boundaries
        ends = boundaries + [signals.duration_ms]

        chapters: list[Chapter] = []
        visual_items = (
            list(signals.chyron_ocr)
            + list(signals.visual_captions)
            + list(signals.swt_timeframes)
        )
        for s, e in zip(starts, ends):
            if e <= s:
                continue
            if self.skip_titling:
                title = ""
            else:
                chapter_asr = format_timed_items_for_prompt(
                    _items_in_range(signals.asr, s, e)
                )
                chapter_visual = format_timed_items_for_prompt(
                    _items_in_range(visual_items, s, e)
                )
                title_result = self.title_chapter(
                    chapter_start_ms=s,
                    chapter_end_ms=e,
                    asr_text=chapter_asr or "[no ASR]",
                    visual_text=chapter_visual or "[no visual]",
                )
                title = (getattr(title_result, "title", "") or "").strip()
            chapters.append(Chapter(start=s, end=e, title=title))
        return chapters

    def _clean_boundaries(self, raw, duration_ms: int) -> list[int]:
        """Clamp, sort, dedup, enforce min spacing. Greedy thinning so the
        earliest acceptable boundary wins when two are too close."""
        cleaned: list[int] = []
        for b in raw:
            try:
                v = int(b)
            except (TypeError, ValueError):
                continue
            v = max(0, min(duration_ms, v))
            if 0 < v < duration_ms:
                cleaned.append(v)
        if not cleaned:
            return []
        cleaned = sorted(set(cleaned))
        if self.min_chapter_duration_ms <= 0:
            return cleaned
        kept = [cleaned[0]]
        for b in cleaned[1:]:
            if b - kept[-1] >= self.min_chapter_duration_ms:
                kept.append(b)
        # Also require the final chapter to be at least min_duration long.
        while kept and (duration_ms - kept[-1]) < self.min_chapter_duration_ms:
            kept.pop()
        return kept


# ----- Interleaved-event, line-index recipe ----------------------------


class BoundaryDetectionByLineIndex(dspy.Signature):
    __doc__ = INTERLEAVED_BOUNDARY_INSTRUCTION

    duration_ms: int = dspy.InputField(desc="total video duration in ms")
    events: str = dspy.InputField(
        desc="numbered, time-sorted multimodal event stream; one event per line"
    )
    boundary_lines: list[int] = dspy.OutputField(
        desc="line numbers (from the events field) where a new chapter begins; "
             "do NOT include 1 (the first chapter is implicit)"
    )


class M5InterleavedRecipe(dspy.Module):
    """Chapter boundary detection over an interleaved ASR+visual+speaker+pause
    event stream. The model emits boundary LINE NUMBERS (not millisecond
    timestamps); we translate each line number back to its underlying time
    via the line-to-ms map we built when formatting the events.

    Title generation is decoupled (same per-chapter ChapterTitling pass as
    M5BoundariesRecipe). Set `skip_titling=True` to return empty titles.
    """

    def __init__(
        self,
        min_chapter_duration_ms: int = DEFAULT_MIN_CHAPTER_DURATION_MS,
        snap_to_shot_window_ms: int = DEFAULT_SNAP_TO_SHOT_WINDOW_MS,
        skip_titling: bool = False,
        include_pauses: bool = True,
        pause_min_gap_ms: int = 1500,
    ):
        super().__init__()
        self.detect_boundaries = dspy.Predict(BoundaryDetectionByLineIndex)
        self.title_chapter = dspy.Predict(ChapterTitling)
        self.min_chapter_duration_ms = max(0, int(min_chapter_duration_ms))
        self.snap_to_shot_window_ms = max(0, int(snap_to_shot_window_ms))
        self.skip_titling = bool(skip_titling)
        self.include_pauses = bool(include_pauses)
        self.pause_min_gap_ms = int(pause_min_gap_ms)

    def forward(self, signals: VideoSignals) -> list[Chapter]:
        if signals.duration_ms <= 0:
            return []

        compacted_captions = compact_captions(
            list(signals.chyron_ocr) + list(signals.visual_captions)
        )
        events_text, line_to_ms = format_events_interleaved(
            asr=signals.asr,
            visual_captions=compacted_captions,
            speaker_segments=signals.speaker_segments,
            acoustic_events=signals.acoustic_events,
            include_pauses=self.include_pauses,
            pause_min_gap_ms=self.pause_min_gap_ms,
        )

        result = self.detect_boundaries(
            duration_ms=signals.duration_ms,
            events=events_text,
        )
        raw_lines = getattr(result, "boundary_lines", []) or []

        # Translate line indices -> ms. Drop indices outside the valid
        # range and the line-1 (implicit start) entry if the model emitted it.
        max_line = len(line_to_ms) - 1
        boundary_ms: list[int] = []
        for v in raw_lines:
            try:
                line = int(v)
            except (TypeError, ValueError):
                continue
            if line <= 1 or line > max_line:
                continue
            boundary_ms.append(line_to_ms[line])

        boundary_ms = self._clean_boundaries(boundary_ms, signals.duration_ms)

        shot_change_times = [it.start_ms for it in signals.swt_timeframes]
        if self.snap_to_shot_window_ms > 0 and shot_change_times and boundary_ms:
            snapped = snap_boundaries_to_shots(
                boundary_ms, shot_change_times,
                tolerance_ms=self.snap_to_shot_window_ms,
            )
            boundary_ms = self._clean_boundaries(snapped, signals.duration_ms)

        starts = [0] + boundary_ms
        ends = boundary_ms + [signals.duration_ms]

        chapters: list[Chapter] = []
        visual_items = list(signals.chyron_ocr) + list(signals.visual_captions)
        for s, e in zip(starts, ends):
            if e <= s:
                continue
            if self.skip_titling:
                title = ""
            else:
                chapter_asr = format_timed_items_for_prompt(
                    _items_in_range(signals.asr, s, e)
                )
                chapter_visual = format_timed_items_for_prompt(
                    _items_in_range(visual_items, s, e)
                )
                title_result = self.title_chapter(
                    chapter_start_ms=s,
                    chapter_end_ms=e,
                    asr_text=chapter_asr or "[no ASR]",
                    visual_text=chapter_visual or "[no visual]",
                )
                title = (getattr(title_result, "title", "") or "").strip()
            chapters.append(Chapter(start=s, end=e, title=title))
        return chapters

    def _clean_boundaries(self, raw, duration_ms: int) -> list[int]:
        cleaned = sorted({max(0, min(duration_ms, int(b))) for b in raw if 0 < int(b) < duration_ms})
        if not cleaned or self.min_chapter_duration_ms <= 0:
            return cleaned
        kept = [cleaned[0]]
        for b in cleaned[1:]:
            if b - kept[-1] >= self.min_chapter_duration_ms:
                kept.append(b)
        while kept and (duration_ms - kept[-1]) < self.min_chapter_duration_ms:
            kept.pop()
        return kept
