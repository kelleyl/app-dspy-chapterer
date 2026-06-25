"""Extract VideoSignals from a MMIF.

Looks for ASR views from VibeVoice or Parakeet (TextDocument + TimeFrame +
Span + Alignment), and optionally a TransNet shots view for shot-change
TimeFrames."""

from __future__ import annotations

from typing import Iterable, Optional

from mmif import Mmif

from .types import TimedItem, VideoSignals


DEFAULT_FPS = 29.97  # NTSC; most FuzzyMemoriesTV archival is 29.97 or 30


def _normalize_app(uri: str) -> str:
    return (uri or "").lower()


def _id(s: str) -> str:
    """Strip view prefix from an annotation id (e.g. ``v_1:t_3`` -> ``t_3``)."""
    return s.split(":", 1)[-1] if ":" in s else s


def _view_time_unit(view, type_substring: str) -> Optional[str]:
    """Read the declared timeUnit for a given MMIF type in a view's contains
    metadata. Returns None if not declared. `type_substring` is matched against
    the type URI (e.g. "TimeFrame"). Lowercased."""
    contains = (getattr(view.metadata, "contains", None) or {})
    try:
        items = contains.items()
    except AttributeError:
        items = []
    for type_uri, type_meta in items:
        if type_substring in str(type_uri):
            try:
                unit = type_meta.get("timeUnit") if hasattr(type_meta, "get") else type_meta["timeUnit"]
            except (KeyError, TypeError):
                unit = None
            return unit.lower() if isinstance(unit, str) else None
    return None


def _video_fps(mmif: Mmif, default: float = DEFAULT_FPS) -> float:
    """Try to read fps from the VideoDocument properties. Falls back to default."""
    for doc in mmif.documents:
        if "VideoDocument" not in str(doc.at_type):
            continue
        props = doc.properties
        for key in ("fps", "frameRate", "frame_rate"):
            val = props.get(key)
            if val is None:
                continue
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return default


def _unwrap_text(val) -> str:
    """Return the inner string of a MMIF text field, regardless of shape.

    MMIF TextDocument `text` is sometimes a plain string, sometimes a dict
    `{"@value": "...", "@language": ...}`, and sometimes a mmif-python
    `Text` wrapper whose `.value` attribute holds the string."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    # mmif-python Text wrapper exposes the inner string as `.value`.
    inner = getattr(val, "value", None)
    if isinstance(inner, str):
        return inner
    # Fall back to dict-style lookup for raw {"@value": "..."}.
    try:
        inner = val.get("@value")
        if isinstance(inner, str):
            return inner
    except (AttributeError, TypeError):
        pass
    return str(val)


def _to_ms(value, unit: Optional[str], fps: float) -> int:
    """Convert a time value to milliseconds given its unit and the video fps."""
    if value is None:
        return 0
    v = float(value)
    u = (unit or "").lower()
    if u in ("frame", "frames"):
        return int(round(v * 1000.0 / fps))
    if u in ("second", "seconds", "s"):
        return int(round(v * 1000.0))
    # Default and "millisecond"/"milliseconds"/"ms": assume already ms.
    return int(round(v))


def _extract_asr_from_view(view, fps: float) -> list[TimedItem]:
    """Pull (start_ms, end_ms, text) triples from an ASR view.

    Supports two common shapes:
      * VibeVoice-style: TextDocument + TimeFrame + Span + Alignment(TF↔Span).
        The Span holds char offsets into the TextDocument.
      * Parakeet-style:  TextDocument + TimeFrame + Token + Sentence +
        Alignment(TF↔Token). The Sentence holds the final text and a list of
        Token targets; per-token timing comes from Token↔TimeFrame alignments.
        We emit one TimedItem per Sentence covering [min token start, max end].

    Reads the view's declared timeUnit for TimeFrame and converts to ms.
    """
    tf_unit = _view_time_unit(view, "TimeFrame")
    text_blob = ""
    tfs: dict[str, tuple[int, int]] = {}
    spans: dict[str, tuple[int, int]] = {}
    tokens: dict[str, tuple[int, int]] = {}
    sentences: list[dict] = []
    aligns: dict[str, str] = {}

    for ann in view.annotations:
        atype = str(ann.at_type)
        props = ann.properties
        if "TextDocument" in atype:
            unwrapped = _unwrap_text(props.get("text"))
            if unwrapped:
                text_blob = unwrapped
        elif "TimeFrame" in atype:
            try:
                tfs[_id(ann.id)] = (_to_ms(props["start"], tf_unit, fps),
                                    _to_ms(props["end"], tf_unit, fps))
            except (KeyError, TypeError, ValueError):
                continue
        elif atype.endswith("/Span") or "Span" in atype:
            # Spans index into a TextDocument by char offset — these stay raw.
            try:
                spans[_id(ann.id)] = (int(props["start"]), int(props["end"]))
            except (KeyError, TypeError, ValueError):
                continue
        elif "Token" in atype:
            # Tokens index into a TextDocument by char offset — these stay raw.
            try:
                tokens[_id(ann.id)] = (int(props["start"]), int(props["end"]))
            except (KeyError, TypeError, ValueError):
                continue
        elif "Sentence" in atype:
            sentences.append({
                "id": _id(ann.id),
                "text": props.get("text", "") or "",
                "targets": [_id(str(t)) for t in props.get("targets", []) or []],
            })
        elif "Alignment" in atype:
            src = props.get("source")
            tgt = props.get("target")
            if src and tgt:
                aligns[_id(str(src))] = _id(str(tgt))
                aligns[_id(str(tgt))] = _id(str(src))

    items: list[TimedItem] = []

    # Parakeet-style: aggregate Tokens into Sentences using time bounds.
    if sentences and tokens:
        for sent in sentences:
            ts = []
            for tok_id in sent["targets"]:
                tf_id = aligns.get(tok_id)
                if tf_id and tf_id in tfs:
                    ts.append(tfs[tf_id])
            if not ts:
                continue
            start = min(t[0] for t in ts)
            end = max(t[1] for t in ts)
            text = (sent["text"] or "").strip()
            if not text:
                continue
            items.append(TimedItem(start_ms=start, end_ms=end, text=" ".join(text.split())))
        items.sort(key=lambda x: (x.start_ms, x.end_ms))
        return items

    # VibeVoice-style: TimeFrame ↔ Span ↔ TextDocument slicing.
    for tf_id, (start, end) in tfs.items():
        target = aligns.get(tf_id)
        if not target:
            continue
        span = spans.get(target)
        if not span:
            continue
        s, e = span
        raw = (text_blob[s:e] if text_blob else "").strip()
        if not raw:
            continue
        items.append(TimedItem(start_ms=start, end_ms=end, text=" ".join(raw.split())))
    items.sort(key=lambda x: (x.start_ms, x.end_ms))
    return items


def extract_asr(mmif: Mmif, fps: float,
                asr_view_match: Optional[Iterable[str]] = None) -> list[TimedItem]:
    """Find the first ASR view whose app URI contains any of `asr_view_match`
    substrings (case-insensitive). Defaults: vibevoice, parakeet, whisper."""
    matches = asr_view_match or ("vibevoice", "parakeet", "whisper")
    for view in mmif.views:
        app = _normalize_app(view.metadata.app or "")
        if any(m in app for m in matches):
            items = _extract_asr_from_view(view, fps)
            if items:
                return items
    return []


def extract_shots(mmif: Mmif, fps: float) -> list[TimedItem]:
    """Find TimeFrames from a transnet shot-detection view, converting from
    the view's declared timeUnit (frame|second|ms) to milliseconds."""
    out: list[TimedItem] = []
    for view in mmif.views:
        app = _normalize_app(view.metadata.app or "")
        if "transnet" not in app and "shot" not in app:
            continue
        unit = _view_time_unit(view, "TimeFrame")
        for ann in view.annotations:
            atype = str(ann.at_type)
            if "TimeFrame" not in atype:
                continue
            try:
                start = _to_ms(ann.properties["start"], unit, fps)
                end = _to_ms(ann.properties["end"], unit, fps)
            except (KeyError, TypeError, ValueError):
                continue
            label = ann.properties.get("label") or ann.properties.get("frameType") or "shot change"
            out.append(TimedItem(start_ms=start, end_ms=end, text=str(label)))
    out.sort(key=lambda x: x.start_ms)
    return out


def extract_visual_text(mmif: Mmif, fps: float) -> list[TimedItem]:
    """Pull per-shot captions / chyron OCR / slate OCR from any captioner
    or OCR view. Handles two layouts:

    1. **Self-contained** (older OCR-style views): annotations carry both
       the timeframe and the text in the same TimeFrame/TimePoint.
    2. **Aligned** (qwen3vl-captioner / smolvlm2-captioner): the view emits
       only TextDocuments + Alignments; the time comes from a TimeFrame
       (or TimePoint) in an upstream view (TransNet shots, SWT, etc.) that
       the Alignment points at. This is the modern pattern.
    """
    # First pass: collect every TimeFrame/TimePoint across all views into
    # a global lookup. Convert each one to ms using its view's declared
    # timeUnit.
    tf_index: dict[str, tuple[int, int]] = {}
    for view in mmif.views:
        tf_unit = _view_time_unit(view, "TimeFrame")
        tp_unit = _view_time_unit(view, "TimePoint")
        for ann in view.annotations:
            atype = str(ann.at_type)
            if "TimeFrame" not in atype and "TimePoint" not in atype:
                continue
            unit = tp_unit if "TimePoint" in atype else tf_unit
            try:
                start = _to_ms(ann.properties["start"], unit, fps)
                end = _to_ms(ann.properties.get("end", ann.properties["start"]), unit, fps)
            except (KeyError, TypeError, ValueError):
                continue
            tf_index[_id(ann.id)] = (start, end)

    out: list[TimedItem] = []
    for view in mmif.views:
        app = _normalize_app(view.metadata.app or "")
        if not any(m in app for m in ("captioner", "ocr", "qwen3vl", "smolvlm")):
            continue
        tf_unit = _view_time_unit(view, "TimeFrame")
        tp_unit = _view_time_unit(view, "TimePoint")

        # Build a local map of TextDocument id → text for this view.
        td_text: dict[str, str] = {}
        for ann in view.annotations:
            if "TextDocument" not in str(ann.at_type):
                continue
            text = _unwrap_text(ann.properties.get("text")).strip()
            if text:
                td_text[_id(ann.id)] = text

        for ann in view.annotations:
            atype = str(ann.at_type)

            # Layout 1: TimeFrame/TimePoint carries text directly.
            if "TimeFrame" in atype or "TimePoint" in atype:
                props = ann.properties
                unit = tp_unit if "TimePoint" in atype else tf_unit
                try:
                    start = _to_ms(props["start"], unit, fps)
                    end = _to_ms(props.get("end", props["start"]), unit, fps)
                except (KeyError, TypeError, ValueError):
                    continue
                text = (_unwrap_text(props.get("text"))
                        or _unwrap_text(props.get("transcription"))
                        or _unwrap_text(props.get("label"))).strip()
                if not text:
                    continue
                out.append(TimedItem(start_ms=start, end_ms=end, text=text))
                continue

            # Layout 2: Alignment(source=upstream TimeFrame, target=local TextDocument).
            if "Alignment" in atype:
                props = ann.properties
                src = _id(str(props.get("source", "")))
                tgt = _id(str(props.get("target", "")))
                # Try both directions: source could be the TimeFrame and
                # target the TextDocument, or vice versa.
                for tf_id, td_id in ((src, tgt), (tgt, src)):
                    if tf_id in tf_index and td_id in td_text:
                        start, end = tf_index[tf_id]
                        out.append(TimedItem(start_ms=start, end_ms=end, text=td_text[td_id]))
                        break

    out.sort(key=lambda x: (x.start_ms, x.end_ms))
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
    fps: Optional[float] = None,
) -> VideoSignals:
    """Build a VideoSignals from a MMIF.

    fps: video frame rate, used to convert any frame-unit TimeFrames to ms.
         If None, read from the VideoDocument (fps/frameRate/frame_rate
         property) or fall back to DEFAULT_FPS (29.97).
    """
    f = fps if fps is not None else _video_fps(mmif)
    asr = extract_asr(mmif, f)
    dur = duration_ms or video_duration_ms(mmif) or (asr[-1].end_ms if asr else 0)
    shots = extract_shots(mmif, f) if (include_shots or include_visual) else []
    visual = extract_visual_text(mmif, f) if include_visual else []
    return VideoSignals(
        duration_ms=dur,
        asr=asr,
        swt_timeframes=shots,
        chyron_ocr=[],
        visual_captions=visual,
    )
