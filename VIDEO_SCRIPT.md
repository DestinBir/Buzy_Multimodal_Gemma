# Buzy AI — Demo Video Script (3 minutes)

---

## [0:00–0:30] The Problem

**Visual:** Split screen — scattered papers, PDF icons, audio files on one side; frustrated person searching through documents on the other.

**Narrator:**
> Every day across Africa, organizations generate contracts, invoices, meeting recordings, and reports — scattered across paper, PDFs, images, and audio files.
>
> When a manager needs a simple answer — "Which supplier is the highest risk?" or "Why did we exceed budget?" — it takes hours of manual cross-referencing.
>
> In the DRC, 80% of small businesses have no document management system. Critical knowledge is lost or buried in silos.

---

## [0:30–1:00] The Solution

**Visual:** Logo animation — Buzy AI. Then show the Streamlit app interface with the file upload area and question box.

**Narrator:**
> Buzy AI solves this. An AI-powered document intelligence system built on Google's Gemma 4 model.
>
> Upload your documents — PDF contracts, scanned invoices, meeting audio, or screenshots. Ask a business question in English or French.
>
> Buzy AI runs them through five stages: intelligent OCR for scanned documents, TF-IDF knowledge base retrieval, a 5-tool AI agent, Gemma 4 reasoning, and a robust JSON extractor. You get a structured, source-attributed report — in seconds.

---

## [1:00–1:45] Live Demo

**Visual:** Screen recording of the Streamlit app. Upload a contract PDF, ask a question, click Analyze.

**Narrator:**
> Let me show you. Here's Buzy AI running on Hugging Face Spaces.
>
> I'll upload a contract and ask: "What are the key obligations and risks?"
>
> *(click Analyze)*
>
> The model loads once in 4-bit quantization using NF4 double quantization — fitting Gemma 4 on just 4 GB of VRAM.
>
> *(results appear)*
>
> Within seconds we get evidence from the source document, explicit reasoning, a confidence score, business impact, and recommended actions. Every claim is cited back to the exact document.
>
> Now let's try a scanned invoice image — Gemma 4's Vision-Language capability reads it directly, extracting structured data like amounts, dates, and supplier names from the raw image.

---

## [1:45–2:15] How It Works Under the Hood

**Visual:** Architecture diagram animation — documents → OCR preprocessing → TF-IDF RAG → Agent tools → Gemma 4 + LoRA → JSON extractor.

**Narrator:**
> Under the hood, we fine-tuned Gemma 4 with LoRA — only 0.25% of parameters trained, taking just 57 seconds on Kaggle GPUs.
>
> We added a TF-IDF retrieval system — lightweight, no vector database needed — that scores document chunks against the question and injects the top results as grounded context.
>
> The AI agent can call five business tools: search the knowledge base, look up supplier info, calculate totals or VAT, convert currencies between XAF, CDF, NGN, and KES, or check the current date for deadline calculations.
>
> We also fixed the classic Markdown vs JSON conflict — the prompt now uses a flat format with strict JSON-only instructions, and a robust extractor strips code fences, trailing commas, and single quotes before parsing.

---

## [2:15–2:45] Why This Matters for Local Impact

**Visual:** Map of DRC / Africa. Photos of small businesses, cooperatives, markets. Then show the dataset preview.

**Narrator:**
> Buzy AI is designed for the African context. French and English bilingual, local-first — runs on a laptop with no internet.
>
> The 100-example synthetic dataset included in the repository covers 12 business domains across 25 African cities: mobile money agents in Kinshasa, cocoa cooperatives in Côte d'Ivoire, cobalt miners in Lubumbashi, and mini-grid solar in rural Tanzania.
>
> A cooperative in Bukavu can upload supplier contracts, record meeting audio, photograph receipts — and get the same structured intelligence that a multinational gets from expensive ERP systems.

---

## [2:45–3:00] Call to Action

**Visual:** Final screen with links: GitHub, Hugging Face Space, Kaggle Notebook.

**Narrator:**
> Buzy AI is open source. Try the live demo, explore the code, or retrain it on your own data.
>
> All links are in the description — GitHub, Hugging Face, and the Kaggle notebook.
>
> Built with Gemma 4 by Google. Merci. Thank you.

---

## End Screen Text

```
Try Buzy AI:
🔗 huggingface.co/spaces/DestinBir/buzy-ai
🔗 github.com/DestinBir/Buzy_Multimodal_Gemma
🔗 kaggle.com/code/destinbir1/buzy-ai-gemma4-lora

Dataset: 100 synthetic African business examples
Tech: Gemma 4 + LoRA 4-bit | TF-IDF RAG | 5-tool Agent | OCR

Built with ❤️ using Gemma 4 by Google
GDG on Campus UCB — Build with Gemma 2026
```
