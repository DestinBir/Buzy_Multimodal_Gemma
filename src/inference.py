import json
import re
import time
import traceback

from src.agent import Agent
from src.loaders import LoadedContext, load_images, load_documents, load_audio
from src.model import load_models, run_gemma
from src.prompt import build_prompt
from src.rag import KnowledgeBase


def _extract_json(text: str) -> str:
    text = text.strip()
    fence_pattern = re.compile(
        r"```(?:json)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE
    )
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        return text[brace_start : brace_end + 1]
    return text


def _parse_json_output(text: str) -> dict:
    cleaned = _extract_json(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    try:
        cleaned = re.sub(r",\s*}", "}", cleaned)
        cleaned = re.sub(r",\s*\]", "]", cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    try:
        escaped = cleaned.replace("'", '"')
        return json.loads(escaped)
    except json.JSONDecodeError:
        pass
    raise json.JSONDecodeError("Could not parse model output as JSON", cleaned, 0)


def format_error(title: str, e: Exception) -> str:
    return (
        f"# {title}\n\n"
        f"```\n{e}\n```\n\n"
        "<details><summary>Traceback</summary>\n\n```\n"
        f"{traceback.format_exc()}\n```\n</details>"
    )


def Buzy_inference(images, documents, audios, question, progress=None):
    start = time.time()
    ctx = LoadedContext()

    try:
        if progress:
            progress(0.05, "Loading images...")
        ctx.images, ctx.image_labels = load_images(images)

        if progress:
            progress(0.20, "Loading & OCRing documents...")
        ctx.documents_text, ctx.doc_sources, ctx.doc_chunks = load_documents(documents)

        bundle = load_models()

        if progress:
            progress(0.40, "Transcribing audio...")
        ctx.audio_text, ctx.audio_sources = load_audio(audios, bundle.asr_pipe)

        if not ctx.images and not ctx.documents_text and not ctx.audio_text and not question:
            return (
                "# Nothing to analyze yet\n\n"
                "Upload at least one document, image, or audio file, "
                "or ask a question, then click **Analyze**."
            ), None

        if progress:
            progress(0.55, "Building knowledge base...")
        kb = KnowledgeBase()
        kb.build(ctx)
        agent = Agent(kb)

        if progress:
            progress(0.60, "Retrieving relevant facts...")
        retrieved = kb.retrieve(question, top_k=6)

        if progress:
            progress(0.65, "Thinking...")
        messages = build_prompt(question, ctx, retrieved_facts=retrieved)
        result = run_gemma(messages)

        tool_name, tool_args = agent.parse_tool_call(result)
        if tool_name:
            if progress:
                progress(0.75, f"Running tool: {tool_name}...")
            tool_output = agent.run_tool(tool_name, tool_args)
            messages = build_prompt(
                question, ctx, retrieved_facts=retrieved, tool_result=tool_output
            )
            result = run_gemma(messages)

        if progress:
            progress(0.85, "Formatting response...")

        try:
            parsed = _parse_json_output(result)
            report = _format_json_report(parsed, ctx, start)
        except json.JSONDecodeError:
            cleaned = _extract_json(result)
            report = cleaned

        if progress:
            progress(1.0, "Done")
        return report, (ctx.images if ctx.images else None)

    except Exception as e:
        return format_error("Analysis failed", e), None


def _format_json_report(parsed: dict, ctx: LoadedContext, start: float) -> str:
    lines = []
    lines.append("# Executive Summary")
    lines.append(parsed.get("business_impact", "No summary available."))
    lines.append("")

    if parsed.get("evidence"):
        lines.append("## Evidence")
        for e in parsed["evidence"]:
            lines.append(f"- {e}")
        lines.append("")

    if parsed.get("reasoning"):
        lines.append("## Reasoning")
        for r in parsed["reasoning"]:
            lines.append(f"- {r}")
        lines.append("")

    if parsed.get("recommended_actions"):
        lines.append("## Recommended Actions")
        for a in parsed["recommended_actions"]:
            lines.append(f"- {a}")
        lines.append("")

    conf = parsed.get("confidence", 0)
    lines.append("## Confidence")
    level = "High" if conf >= 0.7 else "Medium" if conf >= 0.4 else "Low"
    lines.append(f"**{level}** ({conf:.2f})")
    lines.append("")

    elapsed = time.time() - start
    sources = []
    if ctx.doc_sources:
        sources.append("Documents: " + ", ".join(ctx.doc_sources))
    if ctx.image_labels:
        sources.append("Images: " + ", ".join(ctx.image_labels))
    if ctx.audio_sources:
        sources.append("Audio: " + ", ".join(ctx.audio_sources))
    if sources:
        lines.append("---")
        lines.append("**Sources used:** " + "; ".join(sources))
    lines.append(f"*Analysis generated in {elapsed:.1f}s*")

    return "\n".join(lines)
