"""DSPy Chapterer — generates an ordered list of broadcast chapters from
ASR + optional visual signals using a single-shot LLM call. The default
chaptering instruction was discovered by MIPROv2 (instruction-only) over
silver chapter data and improves F1@5s by ~0.05 over a zero-shot prompt
on the cas-2024 gold-23 test set when deployed on a 27B model.

The app calls an external OpenAI-compatible chat-completions endpoint
(vLLM, Ollama, etc.) — set via --apiUrl and --modelName parameters.
"""

import argparse
import logging

import dspy
from clams import ClamsApp, Restifier
from mmif import AnnotationTypes, DocumentTypes, Mmif

from chaptering.prompts import BASELINE_INSTRUCTION, OPTIMIZED_INSTRUCTION
from chaptering.recipe import (
    ChapterGeneration,
    DEFAULT_MIN_CHAPTER_DURATION_MS,
    DEFAULT_SNAP_TO_SHOT_WINDOW_MS,
    M5GenerateRecipe,
)
from chaptering.signals import build_signals


class DspyChapterer(ClamsApp):

    def __init__(self):
        super().__init__()

    def _appmetadata(self):
        # Defined in metadata.py
        pass

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        cfg = self.get_configuration(**parameters)

        api_url: str = cfg["apiUrl"]
        model_name: str = cfg["modelName"]
        api_key: str = cfg.get("apiKey") or "EMPTY"
        use_optimized: bool = cfg["useOptimizedPrompt"]
        include_shots: bool = cfg["useShots"]
        include_visual: bool = cfg["useVisualText"]
        min_dur_ms: int = int(cfg["minChapterDurationMs"])
        snap_window_ms: int = int(cfg["snapToShotWindowMs"])
        max_tokens: int = int(cfg["maxTokens"])
        temperature: float = float(cfg["temperature"])
        request_timeout: float = float(cfg["requestTimeoutSec"])

        self.logger.info(
            f"DspyChapterer: model={model_name} endpoint={api_url} "
            f"optimized={use_optimized} shots={include_shots} visual={include_visual}"
        )

        lm = dspy.LM(
            f"openai/{model_name}",
            api_base=api_url.rstrip("/"),
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=request_timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        dspy.configure(lm=lm)

        if not use_optimized:
            ChapterGeneration.__doc__ = BASELINE_INSTRUCTION
        else:
            ChapterGeneration.__doc__ = OPTIMIZED_INSTRUCTION

        recipe = M5GenerateRecipe(
            min_chapter_duration_ms=min_dur_ms,
            snap_to_shot_window_ms=snap_window_ms if include_shots else 0,
        )

        signals = build_signals(
            mmif,
            include_visual=include_visual,
            include_shots=include_shots,
        )
        if not signals.asr:
            self.logger.warning("DspyChapterer: no ASR found in input MMIF; emitting empty view")
        if signals.duration_ms <= 0:
            self.logger.warning("DspyChapterer: could not determine video duration; output may be empty")

        chapters = recipe(signals)
        self.logger.info(f"DspyChapterer: generated {len(chapters)} chapters")

        video_id = self._video_document_id(mmif)
        view = mmif.new_view()
        self.sign_view(view, parameters)
        view.new_contain(
            AnnotationTypes.TimeFrame,
            document=video_id,
            timeUnit="milliseconds",
        )
        for c in chapters:
            ann = view.new_annotation(AnnotationTypes.TimeFrame)
            ann.add_property("start", int(c.start))
            ann.add_property("end", int(c.end))
            ann.add_property("label", c.title)

        return mmif

    @staticmethod
    def _video_document_id(mmif: Mmif) -> str:
        for doc in mmif.documents:
            if "VideoDocument" in str(doc.at_type):
                return doc.id
        return ""


def get_app():
    return DspyChapterer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="5000", help="port to listen on")
    parser.add_argument("--production", action="store_true", help="run with gunicorn")
    args = parser.parse_args()

    app = get_app()
    http_app = Restifier(app, port=int(args.port))
    if args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
