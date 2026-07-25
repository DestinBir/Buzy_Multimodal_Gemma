# Buzy AI: AI-Powered Document Intelligence for African Organizations

## Problem Statement

Across Africa, small and medium enterprises (SMEs), cooperatives, and NGOs manage critical business decisions using documents scattered across paper, PDFs, images, and audio recordings. Contracts, invoices, meeting minutes, and financial reports exist in isolation: there is no centralized way to extract insights, track risks, or make informed decisions.

In the Democratic Republic of the Congo specifically:
- **80% of SMEs** operate without formal document management systems
- **Manual contract review** costs small businesses significant time and money
- **Meeting decisions** are lost when only recorded as audio: no transcription, no search
- **Multilingual operations** mix French administration with local languages (Lingala, Swahili)
- **Unreliable internet** makes cloud-only solutions impractical

Answering a simple question like *"Which supplier represents the highest operational risk?"* requires hours of cross-referencing documents across multiple formats and languages.

## Solution

**Buzy AI** is a multimodal document intelligence system that transforms raw business documents into structured, explainable decisions using Google's Gemma 4 model.

The system:
1. **Ingests** documents (PDF, DOCX, TXT), images (receipts, contracts, dashboards), and audio (meeting recordings)
2. **Extracts** structured facts using Gemma 4's Vision-Language capabilities
3. **Builds** a lightweight, searchable knowledge base
4. **Retrieves** relevant evidence for each business question
5. **Generates** structured, source-attributed recommendations

### Key Differentiators

- **Source-attributed outputs**: every claim cites the exact document and page
- **Explainable reasoning**: evidence → reasoning → confidence → impact → actions
- **Multimodal**: text, scanned images, and audio in a single pipeline
- **Local-first**: runs on a laptop, no internet required after initial setup
- **Multilingual**: French and English supported; designed for DRC's bilingual context
- **LoRA-efficient**: only 0.25% of parameters trained (12.6M of 5.1B)
- **TF-IDF RAG**: lightweight semantic retrieval without external vector databases
- **5-tool Agent**: search, supplier lookup, currency conversion, calculation, date queries
- **Image preprocessing OCR**: adaptive binarization + contrast enhancement for scanned docs

## How Gemma 4 Was Integrated

Gemma 4 serves as the core intelligence engine across the entire pipeline:

### 1. Base Model
We use **Gemma 4 (E2B-it)** in 4-bit quantization via Unsloth, loaded on consumer GPUs (NVIDIA T4). The model provides:
- Multimodal Vision-Language understanding (reading scanned documents and images)
- Multilingual text generation (French and English)
- Structured JSON output following explicit schemas
- Chain-of-thought reasoning capabilities

### 2. LoRA Fine-Tuning
We fine-tuned Gemma 4 using Low-Rank Adaptation (LoRA, rank 8) on a custom dataset of 4 structured business reasoning examples. Each example trains the model to produce:
- Evidence citations from source documents
- Explicit reasoning chains
- Confidence scoring (0-1)
- Business impact assessment
- Recommended actions

**Training details:**
- Framework: TRL (SFTTrainer)
- Trainable parameters: 12.6M / 5.1B (0.25%)
- Epochs: 20
- Final loss: 0.299
- Hardware: Kaggle (2x NVIDIA T4, ~57 seconds)

### 3. Vision-Language Extraction
Gemma 4's native Vision-Language capabilities are used to extract structured fields from scanned invoice images. We prompt the model with an image and a JSON schema, and it returns structured data (supplier name, invoice ID, amounts, dates).

### 4. Retrieval-Augmented Generation (RAG)
Before answering, the system retrieves relevant facts from the knowledge base using **TF-IDF scoring**: a lightweight retrieval method that works without external vector databases. Query keywords are extracted (French + English stop word removal), scored against document chunks using term frequency and inverse document frequency, and the top-k results are injected into the prompt as grounded context. This ensures Gemma 4 generates evidence-based answers rather than hallucinating.

### 5. AI Agent with Tool Use
The agent extends Gemma 4 with **5 business tools**:
- `search_facts`: keyword search across the knowledge base
- `get_supplier_info`: retrieve all information about a specific supplier
- `calculate`: safe arithmetic evaluation (totals, percentages, VAT)
- `currency_convert`: approximate exchange rates for XAF, XOF, CDF, NGN, KES, GHS, ZAR, and more
- `get_date`: current date/time for deadline calculations

The model outputs a `<tool_call>{"tool": "name", "args": {...}}</tool_call>` tag, the system executes the tool against the knowledge base, and the result is fed back for a second inference pass.

## Technical Architecture

```
User Input (documents, images, audio, question)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
┌───────────────────┐              ┌──────────────────────┐
│  Document Loaders  │              │  Image Preprocessor   │
│  PyMuPDF / DOCX    │              │  Adaptive OCR         │
│  Whisper ASR       │              │  (contrast + sharpen) │
└────────┬──────────┘              └──────────┬───────────┘
         │                                     │
         └──────────────┬──────────────────────┘
                        ▼
┌──────────────────────────────────────────┐
│  Knowledge Base Builder                   │
│  Document chunks → TF-IDF index           │
│  Image labels / Audio transcripts         │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  RAG Retriever (TF-IDF scoring)          │
│  Extract keywords → Score facts → Top-K  │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  Agent (tool orchestration)              │
│  Parse <tool_call> → Execute → Re-run    │
│  (search_facts, calculate, currency,     │
│   get_supplier_info, get_date)           │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  Prompt Builder (JSON-only output)       │
│  No Markdown headers → clean JSON schema │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  Token Limit Guard                       │
│  MAX_PROMPT_TOKENS=4096                  │
│  MAX_NEW_TOKENS=1024                     │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  Gemma 4 + LoRA (4-bit NF4 quant)        │
│  Base: unsloth/gemma-4-E2B-it-4bit       │
│  Adapter: DestinBir/buzy-ai-gemma4-lora  │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  JSON Extractor & Report Formatter       │
│  Strip ``` fences → Fix trailing commas  │
│  → Parse → Render Markdown report        │
└──────────────────────────────────────────┘
```

## Prompt Engineering Strategy

We use a structured prompt template designed to **eliminate the Markdown vs JSON conflict** that plagues many LLM pipelines:

1. **Flat format (no Markdown headers)**: Context sections use plain labels (`Business Question:`, `Uploaded Documents:`, `Tool Result:`) instead of `##` headers, preventing the model from mimicking Markdown in its output
2. **Strict JSON-only instruction**: *"You MUST respond with ONLY a valid JSON object... Do NOT include markdown code blocks, markdown formatting, explanations, or any text before or after the JSON."*
3. **Tool call format**: Explicit `<tool_call>{"tool": "name", "args": {...}}</tool_call>` syntax with tags makes parsing unambiguous
4. **Post-processing robustness**: `_extract_json()` strips ``` fences, finds the outermost `{`/`}`, and fixes trailing commas before parsing

The output schema enforced in training:
```json
{
  "evidence": ["...", "..."],
  "reasoning": ["...", "..."],
  "confidence": 0.82,
  "business_impact": "...",
  "recommended_actions": ["...", "...", "..."]
}
```

## Challenges Encountered

1. **Memory constraints on T4 GPUs**: Gemma 4 is a large model (5B+ parameters). We used 4-bit NF4 quantization via Unsloth (double quant) and gradient accumulation to fit training on 2x T4 GPUs.

2. **Multimodal prompt formatting**: Gemma 4's chat template requires specific formatting for multimodal content. We had to carefully structure messages with alternating image and text content.

3. **Small training dataset**: With only 4 training examples, the model could overfit. We mitigated this by using low-rank adaptation (r=8) and early stopping at 20 epochs.

4. **Markdown vs JSON output conflict**: The model would wrap JSON in ``` fences, add explanatory text around it, or include trailing commas. We solved this with (a) flat-format prompts without Markdown headers, (b) strict JSON-only instructions, and (c) a multi-strategy JSON extractor that handles fences, trailing commas, and single quotes.

5. **Audio integration**: Whisper ASR adds latency. We use lazy loading so the model loads only when audio is uploaded.

6. **OCR reliability on low-quality scans**: African business documents are often photographed on phones rather than scanned. We added adaptive image preprocessing (contrast enhancement, sharpening, binarization) before passing to Tesseract OCR.

## Future Improvements

- **Add Lingala and Swahili support** for broader DRC accessibility
- **Mobile app** for field data collection (photos of receipts, voice notes)
- **Voice interface** for low-literacy users
- **Fine-tune on real African business documents** instead of synthetic data
- **100-example synthetic dataset** now included in `data/african_business_dataset.json` covering 12 business domains across 25 African cities
- **Offline-first PWA** for areas with unreliable internet
- **Integration with local accounting software** (Sage, QuickBooks)
- **Hybrid retrieval**: combine TF-IDF with dense embeddings for better semantic matching

## Expected Impact

Buzy AI can transform how African organizations manage their document workflows:

- **Time savings**: Reduce document analysis from hours to seconds
- **Risk reduction**: Flag contract risks, payment delays, and supplier issues early
- **Knowledge retention**: Convert audio meeting recordings into searchable text
- **Transparency**: Source-attributed outputs build trust in AI recommendations
- **Accessibility**: Runs locally, works in French and English, no expensive ERP needed

## Links

- **GitHub**: https://github.com/DestinBir/Buzy_Multimodal_Gemma
- **Hugging Face Space**: https://huggingface.co/spaces/DestinBir/buzy-ai
- **Fine-tuned Model**: https://huggingface.co/DestinBir/buzy-ai-gemma4-lora
- **Kaggle Notebook**: https://www.kaggle.com/code/destinbir1/buzy-ai-gemma4-lora
- **Demo Video**: [Link to YouTube/Loom]
