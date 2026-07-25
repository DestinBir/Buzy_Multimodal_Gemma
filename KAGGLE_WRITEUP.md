# Buzy AI — AI-Powered Document Intelligence for African Organizations

## Problem Statement

Across Africa, small and medium enterprises (SMEs), cooperatives, and NGOs manage critical business decisions using documents scattered across paper, PDFs, images, and audio recordings. Contracts, invoices, meeting minutes, and financial reports exist in isolation — there is no centralized way to extract insights, track risks, or make informed decisions.

In the Democratic Republic of the Congo specifically:
- **80% of SMEs** operate without formal document management systems
- **Manual contract review** costs small businesses significant time and money
- **Meeting decisions** are lost when only recorded as audio — no transcription, no search
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

- **Source-attributed outputs** — every claim cites the exact document and page
- **Explainable reasoning** — evidence → reasoning → confidence → impact → actions
- **Multimodal** — text, scanned images, and audio in a single pipeline
- **Local-first** — runs on a laptop, no internet required after initial setup
- **Multilingual** — French and English supported; designed for DRC's bilingual context
- **LoRA-efficient** — only 0.25% of parameters trained (12.6M of 5.1B)

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

### 4. Retrieval-Augmented Reasoning
Before answering a business question, the system retrieves relevant facts from the knowledge base by keyword matching. These facts are injected into the prompt as context, enabling Gemma 4 to generate evidence-based answers rather than hallucinating.

### 5. AI Agent with Tool Use
We demonstrate Gemma 4's function-calling capabilities by building a simple agent that can query business tools (`get_invoice_status`, `get_contract_expiry`) and incorporate results into its final answer.

## Technical Architecture

```
User Input (documents, images, audio, question)
        │
        ▼
┌───────────────────┐
│  Document Loaders  │  ← PyMuPDF (PDF), python-docx (DOCX), PIL (images)
│  & Audio ASR       │  ← Whisper (audio transcription)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Prompt Builder    │  ← Multimodal prompt with context + instructions
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Gemma 4 + LoRA   │  ← Fine-tuned for structured business reasoning
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Report Generator  │  ← Structured Markdown with source citations
└───────────────────┘
```

## Prompt Engineering Strategy

We use a structured prompt template with clear sections:

1. **System instructions**: Define the output format (6 Markdown sections)
2. **Context injection**: Uploaded images, document text, and audio transcripts
3. **User question**: The specific business question to answer
4. **Schema enforcement**: Explicit instructions to cite sources and provide confidence

The fine-tuning dataset uses a stricter JSON schema for training:
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

1. **Memory constraints on T4 GPUs**: Gemma 4 is a large model (5B+ parameters). We used 4-bit quantization via Unsloth and gradient accumulation to fit training on 2x T4 GPUs.

2. **Multimodal prompt formatting**: Gemma 4's chat template requires specific formatting for multimodal content. We had to carefully structure messages with alternating image and text content.

3. **Small training dataset**: With only 4 training examples, the model could overfit. We mitigated this by using low-rank adaptation (r=8) and early stopping at 20 epochs.

4. **Audio integration**: Whisper ASR adds latency. We use lazy loading so the model loads only when audio is uploaded.

## Future Improvements

- **Add Lingala and Swahili support** for broader DRC accessibility
- **Mobile app** for field data collection (photos of receipts, voice notes)
- **Voice interface** for low-literacy users
- **Semantic search** replace keyword matching with embeddings
- **Fine-tune on real African business documents** (currently uses synthetic data)
- **Offline-first PWA** for areas with unreliable internet
- **Integration with local accounting software** (Sage, QuickBooks)

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
