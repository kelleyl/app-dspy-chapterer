# DSPy Chapterer

Generates an ordered list of broadcast chapters (start time, end time, title) from a MMIF containing ASR and an optional shot-boundary view. The chaptering instruction was discovered by MIPROv2 prompt optimization over silver chapter data.

## How it works

1. Pulls an ASR view from the input MMIF (VibeVoice, Parakeet, or Whisper).
2. Optionally pulls TransNet shot-boundary TimeFrames and visual-caption text.
3. Calls an external OpenAI-compatible chat-completions endpoint (vLLM, Ollama, etc.) with a single-shot chaptering prompt.
4. Post-processes the LLM output: clamp, sort, drop overlaps, snap each chapter start to the nearest shot change within tolerance, enforce a minimum chapter duration.
5. Emits one MMIF `TimeFrame` per chapter (`timeUnit=milliseconds`); the chapter title is stored on the `label` property.

The default chaptering instruction was found by MIPROv2 (instruction-only, 27B proposer + 9B task model) on a 12-example silver train set. Reported on cas-2024 gold-23 (m5-generate, transcript-only):

| method                                | F1@5s | Pk    |
|---------------------------------------|------:|------:|
| Zero-shot baseline (27B)              | 0.553 | 0.211 |
| MIPROv2-optimized prompt (27B)        | 0.603 | 0.187 |

The 27B improvement transfers from the silver train set to the gold test set; the smaller 9B model regresses slightly under the optimized prompt (cannot reliably follow the more demanding splitting instruction).

## Requirements

- Python 3.10+
- An external OpenAI-compatible chat-completions endpoint serving a 27B-class instruction-following model (recommended). Smaller models work but accuracy drops; the optimized prompt may regress on models below 9B.
- The LLM endpoint must support large prompts (>32K input tokens) for full-length broadcasts.

The app itself does not load any model weights; GPU is only required at the LLM endpoint.

## Parameters

| name                    | type    | default                            | notes |
|-------------------------|---------|------------------------------------|-------|
| `apiUrl`                | string  | `http://localhost:8888/v1`         | OpenAI-compatible endpoint base URL |
| `modelName`             | string  | `Qwen/Qwen3.5-27B-GPTQ-Int4`       | Model identifier served at `apiUrl` |
| `apiKey`                | string  | `EMPTY`                            | Bearer token; `EMPTY` for local vLLM |
| `useOptimizedPrompt`    | boolean | `True`                             | Set False to fall back to zero-shot baseline |
| `useShots`              | boolean | `True`                             | Snap chapter starts to TransNet shot changes |
| `useVisualText`         | boolean | `False`                            | Include OCR / captioner views in the prompt |
| `minChapterDurationMs`  | integer | `5000`                             | Floor on chapter duration (ms) |
| `snapToShotWindowMs`    | integer | `3000`                             | Snap tolerance (ms) |
| `maxTokens`             | integer | `8192`                             | LLM output budget |
| `temperature`           | number  | `0.0`                              | LLM sampling temperature |
| `requestTimeoutSec`     | number  | `600.0`                            | LLM request timeout (s) |

## Running

### Local development

```bash
pip install -r requirements.txt
python3 app.py --port 5000
```

In another shell:

```bash
curl -X POST http://localhost:5000/ \
  -H 'Content-Type: application/json' \
  -d @example.mmif
```

### Docker

```bash
docker build -t app-dspy-chapterer .
docker run --rm --network=host app-dspy-chapterer  # --network=host so it can reach the LLM endpoint at localhost
```

### CLI

```bash
python3 cli.py --apiUrl http://localhost:8888/v1 --modelName Qwen/Qwen3.5-27B-GPTQ-Int4 in.mmif out.mmif
```

## Output

One MMIF `TimeFrame` annotation per chapter, with:

- `start`: int, milliseconds
- `end`: int, milliseconds
- `label`: string, chapter title

The view is signed with all runtime parameters used for the call.
