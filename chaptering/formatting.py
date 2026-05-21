"""Format TimedItems into the `[start-end] text` line representation
the prompt expects."""

from __future__ import annotations

from .types import TimedItem


def format_timed_items_for_prompt(items: list[TimedItem]) -> str:
    """Return one line per item: `[start_ms-end_ms] text`."""
    lines = []
    for it in sorted(items, key=lambda x: (x.start_ms, x.end_ms)):
        text = (it.text or "").strip().replace("\n", " ")
        if not text:
            continue
        lines.append(f"[{int(it.start_ms)}-{int(it.end_ms)}] {text}")
    return "\n".join(lines)
