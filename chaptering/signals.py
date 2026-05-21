"""Extract VideoSignals from a MMIF.

Looks for ASR views from VibeVoice or Parakeet (TextDocument + TimeFrame +
Span + Alignment), and optionally a TransNet shots view for shot-change
TimeFrames."""

from __future__ import annotations

from typing import Iterable, Optional

from mmif import Mmif

from .types import TimedItem, VideoSignals


def _normalize_app(uri: str) -> str:
    return (uri or "").lower()


def _id(s: str) -> str:
    """Strip view prefix from an annotation id (e.g. ``v_1:t_3`` -> ``t_3``)."""
    return s.split(":", 1)[-1] if ":" in s else s


def _extract_asr_from_view(view) -> list[TimedItem]:
    """Pull (start_ms, end_ms, text) triples from an ASR-ish view.

    Expected layout: one TextDocument, one or more TimeFrames with start/end,
    Spans into the text document, and Alignments linking TimeFrame to Span.
    """
    text_blob = ""
    tfs: dict[str, tuple[int, int]] = {}
    spans: dict[str, tuple[int, int]] = {}
    aligns: dict[str, str] = {}

    for ann in view.annotations:
        atype = str(ann.at_type)
        props = ann.properties
        if "TextDocument" in atype:
            text_blob = props.get("text", {}).get("@value", text_blob) or text_blob
        elif "TimeFrame" in atype:
            try:
                tfs[ann.id] = (int(props["start"]), int(props["end"]))
            except (KeyError, TypeError, ValueError):
                continue
        elif atype.endswith("/Span") or "Span" in atype:
            try:
                spans[ann.id] = (int(props["start"]), int(props["end"]))
            except (KeyError, TypeError, ValueError):
                continue
        elif "Alignment" in atype:
            src = props.get("source")
            tgt = props.get("target")
            if src and tgt:
                aligns[_id(str(src))] = _id(str(tgt))
                aligns[_id(str(tgt))] = _id(str(src))

    items: list[TimedItem] = []
    for tf_id, (start, end) in tfs.items():
        target = aligns.get(tf_id) or aligns.get(_id(tf_id))
        if not target:
            continue
        span = spans.get(target) or spans.get(_id(target))
        if not span:
            continue
        s, e = span
        raw = (text_blob[s:e] if text_blob else "").strip()
        if not raw:
            continue
        items.append(TimedItem(start_ms=start, end_ms=end, text=" ".join(raw.split())))
    items.sort(key=lambda x: (x.start_ms, x.end_ms))
    return items


def extract_asr(mmif: Mmif, asr_view_match: Optional[Iterable[str]] = None) -> list[TimedItem]:
    """Find the first ASR view whose app URI contains any of `asr_view_match`
    substrings (case-insensitive). Defaults: vibevoice, parakeet, whisper."""
    matches = asr_view_match or ("vibevoice", "parakeet", "whisper")
    for view in mmif.views:
        app = _normalize_app(view.metadata.app or "")
        if any(m in app for m in matches):
            items = _extract_asr_from_view(view)
            if items:
                return items
    return []


def extract_shots(mmif: Mmif) -> list[TimedItem]:
    """Find TimeFrames from a transnet shot-detection view."""
    out: list[TimedItem] = []
    for view in mmif.views:
        app = _normalize_app(view.metadata.app or "")
        if "transnet" not in app and "shot" not in app:
            continue
        for ann in view.annotations:
            atype = str(ann.at_type)
            if "TimeFrame" not in atype:
                continue
            try:
                start = int(ann.properties["start"])
                end = int(ann.properties["end"])
            except (KeyError, TypeError, ValueError):
                continue
            label = ann.properties.get("label") or ann.properties.get("frameType") or "shot change"
            out.append(TimedItem(start_ms=start, end_ms=end, text=str(label)))
    out.sort(key=lambda x: x.start_ms)
    return out


def extract_visual_text(mmif: Mmif) -> list[TimedItem]:
    """Pull chyron/slate/credits OCR or per-shot caption text from any
    view whose app produces them. Best-effort, missing views just skipped."""
    out: list[TimedItem] = []
    for view in mmif.views:
        app = _normalize_app(view.metadata.app or "")
        if not any(m in app for m in ("captioner", "ocr", "qwen3vl", "smolvlm")):
            continue
        for ann in view.annotations:
            atype = str(ann.at_type)
            if "TextDocument" in atype:
                continue
            if "TimeFrame" not in atype and "TimePoint" not in atype:
                continue
            props = ann.properties
            try:
                start = int(props["start"])
                end = int(props.get("end", start))
            except (KeyError, TypeError, ValueError):
                continue
            text = props.get("text") or props.get("transcription") or props.get("label") or ""
            if isinstance(text, dict):
                text = text.get("@value", "")
            text = str(text or "").strip()
            if not text:
                continue
            out.append(TimedItem(start_ms=start, end_ms=end, text=text))
    out.sort(key=lambda x: x.start_ms)
    return out


def video_duration_ms(mmif: Mmif) -> int:
    """Try to read the video duration. Fall back to max ASR end_ms when
    no duration is recorded on the VideoDocument."""
    for doc in mmif.documents:
        atype = str(doc.at_type)
        if "VideoDocument" not in atype:
            continue
        props = doc.properties
        for key in ("duration", "durationMs", "videoDuration"):
            val = props.get(key)
            if val is None:
                continue
            try:
                return int(float(val))
            except (TypeError, ValueError):
                continue
    return 0


def build_signals(
    mmif: Mmif,
    duration_ms: int = 0,
    include_visual: bool = False,
    include_shots: bool = False,
) -> VideoSignals:
    asr = extract_asr(mmif)
    dur = duration_ms or video_duration_ms(mmif) or (asr[-1].end_ms if asr else 0)
    shots = extract_shots(mmif) if (include_shots or include_visual) else []
    visual = extract_visual_text(mmif) if include_visual else []
    return VideoSignals(
        duration_ms=dur,
        asr=asr,
        swt_timeframes=shots,
        chyron_ocr=[],
        visual_captions=visual,
    )
