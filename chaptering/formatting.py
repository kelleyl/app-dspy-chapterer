"""Format TimedItems into the `[start-end] text` line representation
the prompt expects, plus compaction helpers for noisy caption views."""

from __future__ import annotations

import re

from .types import SpeakerSegment, TimedItem


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


# ----- Interleaved-events format (Chapter-Llama-inspired) ----------------

def _fmt_mmss(ms: int) -> str:
    """Format ms as `m:ss.t` (minutes:seconds.tenths) — compact."""
    total = max(0, int(ms))
    m, rem = divmod(total, 60_000)
    s = rem // 1000
    t = (rem % 1000) // 100
    return f"{m}:{s:02d}.{t}"


def _detect_pauses(
    asr: list[TimedItem],
    min_gap_ms: int = 1500,
) -> list[tuple[int, str]]:
    """Detect silence gaps between consecutive ASR sentences. Returns list
    of (start_ms, message) pairs to inject as `[PAUSE Xs]` events."""
    out = []
    for prev, nxt in zip(asr, asr[1:]):
        gap = nxt.start_ms - prev.end_ms
        if gap >= min_gap_ms:
            out.append((prev.end_ms, f"PAUSE {gap / 1000:.1f}s"))
    return out


def _detect_speaker_changes(
    segments: list[SpeakerSegment],
    min_block_ms: int = 8000,
) -> list[tuple[int, str]]:
    """Emit `speaker change` events on the boundaries between contiguous
    same-speaker blocks. Suppresses events where the OUTGOING block is
    shorter than `min_block_ms` — many diarizers (incl. VibeVoice) over-
    segment, giving every utterance a fresh speaker id; filtering on
    block duration removes that noise while keeping real anchor↔reporter
    transitions."""
    if not segments:
        return []
    # First pass: greedy block-merging by speaker id.
    blocks: list[list[SpeakerSegment]] = [[segments[0]]]
    for s in segments[1:]:
        if s.speaker == blocks[-1][-1].speaker:
            blocks[-1].append(s)
        else:
            blocks.append([s])
    out = []
    for prev, nxt in zip(blocks, blocks[1:]):
        prev_dur = prev[-1].end_ms - prev[0].start_ms
        nxt_dur = nxt[-1].end_ms - nxt[0].start_ms
        if prev_dur < min_block_ms or nxt_dur < min_block_ms:
            continue
        if not prev[-1].speaker or not nxt[0].speaker:
            continue
        out.append((nxt[0].start_ms,
                    f"speaker change: {prev[-1].speaker} -> {nxt[0].speaker}"))
    return out


def format_events_interleaved(
    asr: list[TimedItem],
    visual_captions: list[TimedItem],
    speaker_segments: list[SpeakerSegment] | None = None,
    acoustic_events: list[TimedItem] | None = None,
    include_pauses: bool = True,
    pause_min_gap_ms: int = 1500,
) -> tuple[str, list[int]]:
    """Build one sorted, numbered event stream that interleaves ASR
    sentences, visual captions, speaker changes, pauses, and any acoustic
    event tags.

    Returns:
        (text, line_to_ms): the formatted multi-line string, and a
        parallel list mapping each line index (1-based, matching the
        prefix in the formatted output) to its time in ms. Use this to
        translate model output line indices back to chapter boundary
        times.
    """
    events: list[tuple[int, str, str]] = []  # (start_ms, tag, content)

    for it in asr:
        txt = (it.text or "").strip().replace("\n", " ")
        if txt:
            events.append((it.start_ms, "ASR", txt))

    for it in visual_captions:
        txt = (it.text or "").strip().replace("\n", " ")
        if txt:
            events.append((it.start_ms, "VIS", txt))

    if speaker_segments:
        for s_ms, msg in _detect_speaker_changes(speaker_segments):
            events.append((s_ms, "SPK", msg))

    if acoustic_events:
        for it in acoustic_events:
            txt = (it.text or "").strip()
            if txt:
                events.append((it.start_ms, "AUD", txt))

    if include_pauses and asr:
        for s_ms, msg in _detect_pauses(asr, min_gap_ms=pause_min_gap_ms):
            events.append((s_ms, "PAU", msg))

    events.sort(key=lambda e: (e[0], e[1]))

    lines = []
    line_to_ms: list[int] = [0]  # 1-based, so index 0 is unused
    for i, (ms, tag, content) in enumerate(events, start=1):
        lines.append(f"{i:>4} {_fmt_mmss(ms):>7} {tag} {content}")
        line_to_ms.append(ms)
    return "\n".join(lines), line_to_ms
