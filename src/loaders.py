import os
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image

try:
    import fitz
except ImportError:
    fitz = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None


@dataclass
class LoadedContext:
    images: List[Image.Image] = field(default_factory=list)
    image_labels: List[str] = field(default_factory=list)
    documents_text: str = ""
    doc_sources: List[str] = field(default_factory=list)
    audio_text: str = ""
    audio_sources: List[str] = field(default_factory=list)


def load_images(files):
    images, labels = [], []
    if not files:
        return images, labels
    for f in files:
        path = f.name if hasattr(f, "name") else f
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            labels.append(os.path.basename(path))
        except Exception as e:
            print(f"[Buzy AI] Could not open image {path}: {e}")
    return images, labels


def extract_pdf_text(path: str) -> str:
    if fitz is None:
        return "[PDF extraction unavailable: install pymupdf]"
    text_parts = []
    doc = fitz.open(path)
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text().strip()
        if page_text:
            text_parts.append(f"--- Page {page_num} ---\n{page_text}")
    doc.close()
    return "\n\n".join(text_parts)


def extract_docx_text(path: str) -> str:
    if DocxDocument is None:
        return "[DOCX extraction unavailable: install python-docx]"
    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_txt_text(path: str) -> str:
    with open(path, "r", errors="ignore") as f:
        return f.read()


def load_documents(files):
    if not files:
        return "", []

    combined_blocks = []
    sources = []
    for f in files:
        path = f.name if hasattr(f, "name") else f
        filename = os.path.basename(path)
        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext == ".pdf":
                text = extract_pdf_text(path)
            elif ext == ".docx":
                text = extract_docx_text(path)
            elif ext == ".txt":
                text = extract_txt_text(path)
            else:
                text = f"[Unsupported document type: {ext}]"
        except Exception as e:
            text = f"[Error reading {filename}: {e}]"

        combined_blocks.append(f"### Source: {filename}\n{text}")
        sources.append(filename)

    return "\n\n".join(combined_blocks), sources


def load_audio(files, asr_pipe):
    if not files:
        return "", []

    combined_blocks = []
    sources = []
    for f in files:
        path = f.name if hasattr(f, "name") else f
        filename = os.path.basename(path)
        try:
            result = asr_pipe(path, return_timestamps=True)
            transcript = result["text"].strip()
        except Exception as e:
            transcript = f"[Error transcribing {filename}: {e}]"

        combined_blocks.append(f"### Meeting audio: {filename}\n{transcript}")
        sources.append(filename)

    return "\n\n".join(combined_blocks), sources
