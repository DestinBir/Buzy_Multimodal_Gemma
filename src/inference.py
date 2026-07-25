import time
import traceback

import gradio as gr

from src.model import load_models, run_gemma, MODEL_ID
from src.loaders import LoadedContext, load_images, load_documents, load_audio
from src.prompt import build_prompt


def format_error(title: str, e: Exception) -> str:
    return (
        f"# ⚠️ {title}\n\n"
        f"```\n{e}\n```\n\n"
        "<details><summary>Traceback</summary>\n\n```\n"
        f"{traceback.format_exc()}\n```\n</details>"
    )


def _build_sources_footer(ctx: LoadedContext, elapsed: float) -> str:
    lines = ["**Sources used**"]
    if ctx.doc_sources:
        lines.append("- Documents: " + ", ".join(ctx.doc_sources))
    if ctx.image_labels:
        lines.append("- Images: " + ", ".join(ctx.image_labels))
    if ctx.audio_sources:
        lines.append("- Audio: " + ", ".join(ctx.audio_sources))
    lines.append(f"\n_Analysis generated in {elapsed:.1f}s using `{MODEL_ID}`._")
    return "\n".join(lines)


def Buzy_inference(images, documents, audios, question, progress=gr.Progress()):
    start = time.time()
    ctx = LoadedContext()

    try:
        progress(0.05, desc="Loading images...")
        ctx.images, ctx.image_labels = load_images(images)

        progress(0.20, desc="Extracting document text...")
        ctx.documents_text, ctx.doc_sources = load_documents(documents)

        progress(0.40, desc="Loading models...")
        bundle = load_models()

        progress(0.50, desc="Transcribing audio...")
        ctx.audio_text, ctx.audio_sources = load_audio(audios, bundle.asr_pipe)

        if not ctx.images and not ctx.documents_text and not ctx.audio_text and not question:
            return (
                "# Nothing to analyze yet\n\n"
                "Upload at least one document, image, or audio file, or ask a question, "
                "then click **Analyze**."
            ), None

        progress(0.65, desc="Building multimodal prompt...")
        messages = build_prompt(question, ctx)

        progress(0.75, desc="Running Gemma inference...")
        report = run_gemma(messages)

        elapsed = time.time() - start
        sources_note = _build_sources_footer(ctx, elapsed)
        report = f"{report}\n\n---\n{sources_note}"

        progress(1.0, desc="Done")
        return report, (ctx.images if ctx.images else None)

    except Exception as e:
        return format_error("Analysis failed", e), None
