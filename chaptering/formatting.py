"""Format TimedItems into the `[start-end] text` line representation
the prompt expects, plus compaction helpers for noisy caption views."""

from __future__ import annotations

import re

from .types import TimedItem


# Phrases captioners stick on the front of every description. Stripping
# them shaves 30-50 chars per shot without losing signal.
_CAPTION_PREFIXES = [
    "the video frame shows ",
    "the video frame depicts ",
    "the video frame features ",
    "the video shows ",
    "the image shows ",
    "the image depicts ",
    "the image features ",
    "the frame shows ",
    "the picture shows ",
    "the scene shows ",
    "this video frame shows ",
    "this image shows ",
    "in this video frame, ",
    "in the video frame, ",
]


def _strip_prefix(s: str) -> str:
    low = s.lower()
    for p in _CAPTION_PREFIXES:
        if low.startswith(p):
            return s[len(p):].lstrip()
    return s


_SENT_END = re.compile(r"(?<=[\.!?])\s")


def _first_sentence(s: str, max_chars: int) -> str:
    """First sentence, but cap at max_chars."""
    s = s.strip()
    m = _SENT_END.search(s)
    if m:
        s = s[:m.start()].rstrip(" .!?")
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "..."
    return s


def compact_captions(
    items: list[TimedItem],
    max_chars_per_caption: int = 120,
    dedup_prefix_chars: int = 60,
) -> list[TimedItem]:
    """Strip boilerplate prefixes, keep just the first sentence, and merge
    consecutive captions whose normalized prefix matches.

    Captioners (qwen3vl, smolvlm) produce one verbose caption per shot.
    Long news stories or talking-head segments produce sequences of
    near-identical captions — useful as a 'no change' signal but bloats
    the prompt. We collapse runs into one TimedItem with the combined
    time range, keeping just the first occurrence's text.
    """
    if not items:
        return []

    cleaned: list[TimedItem] = []
    for it in items:
        text = it.text or ""
        text = _strip_prefix(text)
        text = _first_sentence(text, max_chars_per_caption)
        if not text:
            continue
        cleaned.append(TimedItem(start_ms=it.start_ms, end_ms=it.end_ms, text=text))

    if not cleaned:
        return []

    cleaned.sort(key=lambda x: (x.start_ms, x.end_ms))

    merged: list[TimedItem] = [cleaned[0]]
    for it in cleaned[1:]:
        last = merged[-1]
        a = it.text[:dedup_prefix_chars].lower()
        b = last.text[:dedup_prefix_chars].lower()
        if a == b:
            merged[-1] = TimedItem(
                start_ms=last.start_ms,
                end_ms=max(last.end_ms, it.end_ms),
                text=last.text,
            )
        else:
            merged.append(it)
    return merged


def format_timed_items_for_prompt(items: list[TimedItem]) -> str:
    """Return one line per item: `[start_ms-end_ms] text`."""
    lines = []
    for it in sorted(items, key=lambda x: (x.start_ms, x.end_ms)):
        text = (it.text or "").strip().replace("\n", " ")
        if not text:
            continue
        lines.append(f"[{int(it.start_ms)}-{int(it.end_ms)}] {text}")
    return "\n".join(lines)
