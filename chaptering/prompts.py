"""Baseline and MIPROv2-optimized chaptering instructions.

The optimized instruction was produced by running MIPROv2 (instruction-only,
``num_instruct_candidates=6``) on a 12-example silver train set with the
27B model as the proposer and the 9B model as the task model. F1@5s on the
cas-2024 gold-23 test set when the optimized prompt is deployed on the 27B:

    baseline transcript only          F1@5s = 0.553   Pk = 0.211
    MIPROv2-optimized prompt on 27B   F1@5s = 0.603   Pk = 0.187

The optimized instruction is bundled as the runtime default."""

from __future__ import annotations

BASELINE_INSTRUCTION = """Produce an ordered list of chapters covering the broadcast.

You are given the multi-modal signals for a single video (ASR transcript
with timestamps, optionally visual signals, and the total duration). Your
job is to:
  1. Decide the natural broadcast units (segments, commercial breaks,
     station IDs, sponsor billboards, etc.) based on what is in the
     transcript and visuals.
  2. Output an ordered list of chapters with start_ms, end_ms, and a
     short descriptive title.

Constraints:
  - Chapters must be in temporal order: chapter i+1's start_ms >= chapter
    i's end_ms (no overlaps).
  - Together they should cover most of the broadcast: chapter[0].start_ms
    should be near 0 and chapter[-1].end_ms should be near duration_ms.
  - Commercials are typically 15-60s long; news stories typically
    60-300s; opening/closing segments and sponsor billboards are short.
  - Titles should be short (3-15 words), specific, and descriptive of the
    chapter's content.
  - Do NOT generate placeholder titles like "Segment 1" if the actual
    topic is recoverable from the signals."""


OPTIMIZED_INSTRUCTION = """You are an expert video segmentation engine. Your task is to analyze the provided multi-modal signals—specifically the total video duration, a time-stamped ASR transcript, and visual cues (OCR, captions, scene labels)—to generate a precise, ordered list of broadcast chapters.

**Input Data:**
- `duration_ms`: Total length of the video in milliseconds.
- `asr_text`: Full transcript with timestamps (format: `[start_ms-end_ms] Sentence`).
- `visual_text`: Visual signals with timestamps (may be empty).

**Task Steps:**
1. **Identify Segments**: Analyze the narrative flow and visual markers to detect natural broadcast boundaries. Look for transitions between:
   - News stories (typically 60s–300s).
   - Commercial breaks (typically 15s–60s).
   - Station IDs, sponsor billboards, or opening/closing segments (short duration).
   - Sports highlights, weather reports, or feature stories.
2. **Determine Boundaries**: Assign precise `start_ms` and `end_ms` to each segment based on the timestamps in the transcript and visual cues.
3. **Generate Titles**: Create a concise, descriptive title (3–15 words) for each chapter that reflects the specific content (e.g., \"Commercial: Visine Eye Drops\", \"Soviet trawler seized off Massachusetts coast\"). Avoid generic placeholders like \"Segment 1\" unless the content is truly indistinguishable.

**Constraints:**
- **Chronological Order**: Chapters must be strictly sequential. The `start_ms` of chapter `i+1` must be greater than or equal to the `end_ms` of chapter `i`.
- **Full Coverage**: The first chapter must start near 0, and the last chapter must end near `duration_ms`. Minimize gaps between segments.
- **No Overlaps**: Ensure `start_ms` <= `end_ms` for each chapter, and that chapters do not overlap.
- **Title Quality**: Titles must be specific. If the topic is clear from the text or visuals, use it. If a segment is a commercial, identify the product if possible.

**Output Format:**
Return a JSON-like list of objects, where each object contains:
- `start_ms`: (integer) Start time in milliseconds.
- `end_ms`: (integer) End time in milliseconds.
- `title`: (string) Descriptive title.

Example:
```json
[
  {\"start_ms\": 0, \"end_ms\": 3690, \"title\": \"ABC Weekend News Intro\"},
  {\"start_ms\": 3690, \"end_ms\": 218400, \"title\": \"Soviet Trawler Seized Off Massachusetts Coast\"}
]
```

Begin your analysis now based on the provided signals."""
