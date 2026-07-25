from src.loaders import LoadedContext
from src.agent import TOOL_DESCRIPTIONS

JSON_SCHEMA = """{
  "evidence": ["list of specific facts from documents that support the analysis"],
  "reasoning": ["step-by-step logical reasoning connecting evidence to conclusions"],
  "confidence": 0.0-1.0,
  "business_impact": "description of business impact in one sentence",
  "recommended_actions": ["list of specific, actionable recommendations"]
}"""

OUTPUT_FORMAT_INSTRUCTION = (
    "You MUST respond with ONLY a valid JSON object matching this schema:\n"
    f"{JSON_SCHEMA}\n\n"
    "Do NOT include markdown code blocks, markdown formatting, "
    "explanations, or any text before or after the JSON. "
    "Your entire response must be parseable by json.loads(). "
    "If you do not have enough information, set confidence to 0.1 "
    "and evidence to an empty list."
)

AGENT_INSTRUCTIONS = (
    "You are Buzy AI, a business intelligence assistant for African organizations.\n\n"
    "Available tools:\n"
    f"{TOOL_DESCRIPTIONS}\n\n"
    "To call a tool, output ONLY:\n"
    '<tool_call>{"tool": "tool_name", "args": {"param": "value"}}</tool_call>\n\n'
    "Then wait for the result. After receiving the result, "
    "provide your final answer as JSON.\n\n"
    "If no tool is needed, answer directly.\n\n"
    "Rules:\n"
    '- Every claim in "evidence" must cite the specific source document\n'
    '- "confidence" must be a number between 0 and 1\n'
    '- "recommended_actions" must be specific, actionable steps\n'
    "- Use English or French based on the question language\n"
    "- If information is missing, say so explicitly - do not invent facts\n\n"
    f"{OUTPUT_FORMAT_INSTRUCTION}"
)


def build_prompt(question: str, ctx: LoadedContext, retrieved_facts: list = None, tool_result: str = None) -> list:
    content = []
    for img in ctx.images:
        content.append({"type": "image", "image": img})

    sections = []
    if tool_result:
        sections.append("Tool Result:\n" + tool_result)
    sections.append(f"Business Question: {question.strip() if question else '(No question provided)'}")
    if ctx.documents_text.strip():
        sections.append("Uploaded Documents:\n" + ctx.documents_text[:4000])
    if ctx.audio_text.strip():
        sections.append("Meeting Transcript:\n" + ctx.audio_text[:2000])
    if retrieved_facts:
        sections.append("Retrieved Facts:\n" + "\n".join(f"- {f}" for f in retrieved_facts))
    sections.append(AGENT_INSTRUCTIONS)

    content.append({"type": "text", "text": "\n\n".join(sections)})
    return [{"role": "user", "content": content}]
