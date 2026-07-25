from src.loaders import LoadedContext

ANALYSIS_INSTRUCTIONS = """Please provide a structured business analysis with the following sections, in Markdown, using exactly these headers:

# Executive Summary

# Key Findings

# Business Risks

# Recommended Actions

# Evidence

For every claim in "Key Findings", "Business Risks", and "Evidence", cite the specific source it came from (e.g. "Contract.pdf – page 5", "Meeting transcript – Meeting.mp3", "Image – receipt_003.jpg"). If information is missing or uncertain, say so explicitly rather than guessing.

# Confidence

End with a single line: High, Medium, or Low, plus a one-sentence justification.
"""


def build_prompt(question: str, ctx: LoadedContext) -> list:
    content = []

    for img in ctx.images:
        content.append({"type": "image", "image": img})

    text_block = f"""# Question

{question.strip() if question else "(No question provided — give a general business overview of the uploaded material.)"}

# Documents

{ctx.documents_text if ctx.documents_text else "(No documents uploaded.)"}

# Meeting Transcript

{ctx.audio_text if ctx.audio_text else "(No audio uploaded.)"}

{ANALYSIS_INSTRUCTIONS}
"""

    content.append({"type": "text", "text": text_block})

    messages = [{"role": "user", "content": content}]
    return messages
