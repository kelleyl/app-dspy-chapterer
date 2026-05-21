"""App metadata for the DSPy Chapterer.

DO NOT CHANGE the name of the file.
"""

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from mmif import AnnotationTypes, DocumentTypes


# Defaults kept here (not imported from chaptering.recipe) so the metadata
# dump can run without the heavier runtime deps.
DEFAULT_MIN_CHAPTER_DURATION_MS_VALUE = 5000
DEFAULT_SNAP_TO_SHOT_WINDOW_MS_VALUE = 3000


# DO NOT CHANGE the function name
def appmetadata() -> AppMetadata:
    metadata = AppMetadata(
        name="DSPy Chapterer",
        description=(
            "Generates an ordered list of broadcast chapters (start/end + title) "
            "from ASR and optional visual signals using a single-shot LLM call. "
            "The default chaptering instruction was discovered by MIPROv2 prompt "
            "optimization over silver chapter data and improves F1@5s by ~0.05 over "
            "a zero-shot prompt on the cas-2024 gold-23 test set when deployed on "
            "a 27B-class model. Calls an external OpenAI-compatible chat-completions "
            "endpoint (vLLM, Ollama, etc.); the LLM itself is not bundled."
        ),
        app_license="Apache2.0",
        identifier="dspy-chapterer",
        url="https://github.com/kelleyl/app-dspy-chapterer",
        analyzer_version="dspy-3.1.3",
        analyzer_license="MIT",
        est_gpu_mem_min=0,
        est_gpu_mem_typ=0,
    )

    metadata.add_input(DocumentTypes.VideoDocument, required=True)
    metadata.add_input(
        AnnotationTypes.TimeFrame,
        required=True,
        description="ASR TimeFrames from an ASR view (VibeVoice, Parakeet, Whisper) "
                    "with companion TextDocument, Spans, and Alignments.",
    )
    metadata.add_input(
        AnnotationTypes.TimeFrame,
        required=False,
        description="Optional TransNet shot-boundary TimeFrames; used for snap-to-shot postprocessing.",
    )

    metadata.add_output(
        AnnotationTypes.TimeFrame,
        timeUnit="milliseconds",
        description="One TimeFrame per generated chapter. The `label` property holds the chapter title.",
    )

    metadata.add_parameter(
        name="apiUrl",
        type="string",
        default="http://localhost:8888/v1",
        description="Base URL of the OpenAI-compatible chat-completions endpoint serving the chaptering model.",
    )
    metadata.add_parameter(
        name="modelName",
        type="string",
        default="Qwen/Qwen3.5-27B-GPTQ-Int4",
        description="Model identifier served at apiUrl. 27B-class models work best with the optimized prompt.",
    )
    metadata.add_parameter(
        name="apiKey",
        type="string",
        default="EMPTY",
        description="Bearer token for the endpoint. Use `EMPTY` for local vLLM servers without auth.",
    )
    metadata.add_parameter(
        name="useOptimizedPrompt",
        type="boolean",
        default=True,
        description="Use the MIPROv2-optimized chaptering instruction (recommended). "
                    "Set False to fall back to the zero-shot baseline prompt.",
    )
    metadata.add_parameter(
        name="useShots",
        type="boolean",
        default=True,
        description="Snap each chapter's start to the nearest TransNet shot change within "
                    "snapToShotWindowMs, when shot TimeFrames are present.",
    )
    metadata.add_parameter(
        name="useVisualText",
        type="boolean",
        default=False,
        description="Include OCR / visual-caption text from upstream captioner views in the LLM prompt.",
    )
    metadata.add_parameter(
        name="minChapterDurationMs",
        type="integer",
        default=DEFAULT_MIN_CHAPTER_DURATION_MS_VALUE,
        description="Floor on emitted chapter duration in ms; shorter chapters get their end extended.",
    )
    metadata.add_parameter(
        name="snapToShotWindowMs",
        type="integer",
        default=DEFAULT_SNAP_TO_SHOT_WINDOW_MS_VALUE,
        description="Tolerance in ms for snapping each chapter start to the nearest TransNet shot change. "
                    "Only applies when useShots is True.",
    )
    metadata.add_parameter(
        name="maxTokens",
        type="integer",
        default=8192,
        description="Maximum output tokens for the chaptering LLM call.",
    )
    metadata.add_parameter(
        name="temperature",
        type="number",
        default=0.0,
        description="Sampling temperature for the LLM call.",
    )
    metadata.add_parameter(
        name="requestTimeoutSec",
        type="number",
        default=600.0,
        description="Per-request timeout for the LLM endpoint, in seconds.",
    )

    return metadata


if __name__ == "__main__":
    import sys

    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
