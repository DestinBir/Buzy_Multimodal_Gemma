import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image, ImageEnhance, ImageFilter

try:
    import fitz
except ImportError:
    fitz = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

MAX_CHUNK_WORDS = 500
CHUNK_OVERLAP = 50
MAX_AUDIO_WORDS = 800


@dataclass
class LoadedContext:
    images: List[Image.Image] = field(default_factory=list)
    image_labels: List[str] = field(default_factory=list)
    documents_text: str = ""
    doc_sources: List[str] = field(default_factory=list)
    doc_chunks: List[dict] = field(default_factory=list)
    audio_text: str = ""
    audio_sources: List[str] = field(default_factory=list)


def _preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = img.filter(ImageFilter.SHARPEN)
    threshold = 128
    img = img.point(lambda x: 255 if x > threshold else 0, "1")
    return img


def _ocr_image(img: Image.Image) -> Optional[str]:
    if pytesseract is None:
        return None
    try:
        text = pytesseract.image_to_string(img, lang="fra+eng")
        if text.strip():
            return text.strip()
        processed = _preprocess_image_for_ocr(img)
        text = pytesseract.image_to_string(processed, lang="fra+eng")
        return text.strip() or None
    except Exception as e:
        print(f"[Buzy AI] OCR failed: {e}")
        return None


def load_images(files) -> Tuple[List[Image.Image], List[str]]:
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


def ocr_pdf(path: str) -> Optional[str]:
    if fitz is None or pytesseract is None:
        return None
    try:
        doc = fitz.open(path)
        pages = []
        for page_num, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=250)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang="fra+eng")
            if text.strip():
                pages.append(f"--- Page {page_num} (OCR) ---\n{text.strip()}")
        doc.close()
        return "\n\n".join(pages) if pages else None
    except Exception as e:
        print(f"[Buzy AI] PDF OCR failed: {e}")
        return None


def extract_docx_text(path: str) -> str:
    if DocxDocument is None:
        return "[DOCX extraction unavailable: install python-docx]"
    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_txt_text(path: str) -> str:
    with open(path, "r", errors="ignore") as f:
        return f.read()


def chunk_text(
    text: str, source: str, max_words: int = MAX_CHUNK_WORDS, overlap: int = CHUNK_OVERLAP
) -> List[dict]:
    words = text.split()
    if len(words) <= max_words:
        return [{"source": source, "text": text, "chunk_id": 0}]

    chunks = []
    start = 0
    chunk_id = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append({
            "source": source,
            "text": " ".join(words[start:end]),
            "chunk_id": chunk_id,
        })
        chunk_id += 1
        start += max_words - overlap
    return chunks


def load_documents(files) -> Tuple[str, List[str], List[dict]]:
    if not files:
        return "", [], []

    combined_blocks = []
    sources = []
    all_chunks = []

    for f in files:
        path = f.name if hasattr(f, "name") else f
        filename = os.path.basename(path)
        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext == ".pdf":
                text = extract_pdf_text(path)
                if not text.strip() or len(text.strip()) < 50:
                    ocr_text = ocr_pdf(path)
                    if ocr_text:
                        text = ocr_text
            elif ext == ".docx":
                text = extract_docx_text(path)
            elif ext == ".txt":
                text = extract_txt_text(path)
            else:
                text = ""
        except Exception as e:
            text = f"[Error reading {filename}: {e}]"

        combined_blocks.append(f"### Source: {filename}\n{text}")
        sources.append(filename)
        all_chunks.extend(chunk_text(text, filename))

    return "\n\n".join(combined_blocks), sources, all_chunks


def load_audio(files, asr_pipe) -> Tuple[str, List[str]]:
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
            words = transcript.split()
            if len(words) > MAX_AUDIO_WORDS:
                transcript = " ".join(words[:MAX_AUDIO_WORDS])
                transcript += f"\n\n[Transcript truncated to {MAX_AUDIO_WORDS} words]"
        except Exception as e:
            transcript = f"[Error transcribing {filename}: {e}]"

        combined_blocks.append(f"### Meeting audio: {filename}\n{transcript}")
        sources.append(filename)

    return "\n\n".join(combined_blocks), sources
