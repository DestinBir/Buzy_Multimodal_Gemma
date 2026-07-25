# Buzy AI — Demo Video Script (3 minutes)

---

## [0:00–0:30] The Problem

**Visual:** Split screen — left side shows scattered papers, PDF icons, audio files; right side shows a frustrated person searching through documents.

**Narrator:**
> Every day, organizations across Africa generate contracts, invoices, meeting recordings, and reports. This information is scattered across paper, PDFs, images, and audio files.
>
> When a manager needs to answer a simple question — "Which supplier is most reliable?" or "Why did our expenses go over budget?" — it takes hours of manual work.
>
> In the DRC, 80% of small businesses have no document management system at all. Critical knowledge is lost or buried.

---

## [0:30–1:00] The Solution

**Visual:** Logo animation — Buzy AI. Then show the Gradio app interface.

**Narrator:**
> Buzy AI solves this. It's an AI-powered document intelligence system built on Google's Gemma 4 model.
>
> You upload your documents — PDF contracts, scanned invoices, meeting audio recordings, or screenshots of dashboards. Then you ask a business question in plain English or French.
>
> Buzy AI extracts the facts, builds a knowledge base, and generates a structured, source-attributed report — in seconds.

---

## [1:00–1:45] Live Demo

**Visual:** Screen recording of the Gradio app. Upload a contract PDF, type a question, click Analyze.

**Narrator:**
> Let me show you. Here's the Buzy AI app, running on Hugging Face Spaces.
>
> I'll upload a contract PDF, and ask: "What are the key obligations and risks?"
>
> *(click Analyze)*
>
> The model loads — this happens once — and within seconds, we get a full analysis: executive summary, key findings, business risks, recommended actions, and evidence citations pointing back to the exact source document.
>
> Now let's try with an invoice image. I'll upload a scanned invoice and ask for a summary with anomaly flags.
>
> *(upload invoice, click Analyze)*
>
> Gemma 4's Vision-Language capability reads the invoice directly from the image and extracts structured data like amounts, dates, and supplier names.

---

## [1:45–2:15] How It Works Under the Hood

**Visual:** Architecture diagram animation — documents → Gemma 4 → knowledge base → reasoning → recommendations.

**Narrator:**
> Under the hood, we fine-tuned Gemma 4 using LoRA — Low-Rank Adaptation — on a structured reasoning dataset. Only 0.25% of the model's parameters were trained, which took just 57 seconds on Kaggle GPUs.
>
> The model learns to produce evidence-based answers with explicit reasoning chains, confidence scores, and actionable recommendations. Every claim cites the original source document.
>
> We also added Whisper ASR for meeting audio transcription, and built a simple AI agent that can query business tools like invoice status and contract expiry dates.

---

## [2:15–2:45] Why This Matters for Local Impact

**Visual:** Map of DRC / Africa. Photos of small businesses, cooperatives, markets.

**Narrator:**
> Buzy AI is designed for the African context. It works in both French and English, critical for DRC's bilingual environment. It runs on a laptop — no expensive cloud subscription, no reliable internet required after setup.
>
> A small cooperative in Bukavu can upload their supplier contracts, record meeting minutes as audio, photograph receipts — and get the same structured intelligence that a multinational corporation gets from expensive ERP systems.
>
> This is AI for local impact: practical, accessible, and built for real needs.

---

## [2:45–3:00] Call to Action

**Visual:** Final screen with links: GitHub, Hugging Face Space, Kaggle Notebook.

**Narrator:**
> Buzy AI is open source. You can try the live demo, read the code, or run it yourself.
>
> Links to the GitHub repository, Hugging Face Space, and Kaggle notebook are in the description.
>
> Built with Gemma 4 by Google. Thank you.

---

## End Screen Text

```
Try Buzy AI:
🔗 huggingface.co/spaces/DestinBir/buzy-ai
🔗 github.com/DestinBir/Buzy_Multimodal_Gemma
🔗 kaggle.com/code/destinbir1/buzy-ai-gemma4-lora

Built with ❤️ using Gemma 4 by Google
GDG on Campus UCB — Build with Gemma 2026
```
