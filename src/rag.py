import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from src.loaders import LoadedContext


AFRICAN_STOP_WORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "dans", "pour",
    "sur", "avec", "est", "sont", "ce", "cet", "cette", "ces", "qui",
    "que", "quoi", "dont", "ou", "et", "mais", "ni", "car", "sans",
    "the", "a", "an", "of", "in", "to", "for", "with", "on", "at",
    "by", "what", "which", "why", "how", "when", "where", "does",
    "do", "is", "are", "was", "were", "has", "have", "had", "been",
    "being", "it", "its", "this", "that", "these", "those", "not",
    "no", "nor", "so", "if", "then", "than", "too", "very", "just",
    "also", "more", "some", "any", "each", "every", "all", "both",
    "few", "most", "other", "into", "about", "upon", "vs", "via",
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi",
    "dimanche", "janvier", "février", "mars", "avril", "mai",
    "juin", "juillet", "août", "septembre", "octobre", "novembre",
    "décembre",
}


class KnowledgeBase:
    def __init__(self):
        self.facts: List[dict] = []
        self.built = False
        self._idf: Dict[str, float] = {}

    def build(self, ctx: LoadedContext):
        self.facts = []
        for chunk in ctx.doc_chunks:
            self.facts.append({
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "type": "document",
            })
        for label, img in zip(ctx.image_labels, ctx.images):
            self.facts.append({
                "source": label,
                "text": f"[Image: {label}]",
                "type": "image",
            })
        if ctx.audio_text.strip():
            for src in ctx.audio_sources:
                self.facts.append({
                    "source": src,
                    "text": ctx.audio_text,
                    "type": "audio",
                })
        self._compute_idf()
        self.built = True

    def _compute_idf(self):
        n_docs = len(self.facts)
        if n_docs == 0:
            return
        df: Dict[str, int] = {}
        for fact in self.facts:
            tokens = set(self._tokenize(fact["text"]))
            for t in tokens:
                df[t] = df.get(t, 0) + 1
        self._idf = {
            t: math.log((n_docs + 1) / (freq + 1)) + 1
            for t, freq in df.items()
        }

    def retrieve(
        self, question: str, top_k: int = 6, use_tfidf: bool = True
    ) -> List[str]:
        if not self.built or not self.facts:
            return []

        query_tokens = self._tokenize(question)
        if not query_tokens:
            return [f"{fact['source']}: {fact['text'][:600]}" for fact in self.facts[:top_k]]

        scored = []
        for fact in self.facts:
            doc_tokens = self._tokenize(fact["text"])
            doc_counts = Counter(doc_tokens)
            source_tokens = self._tokenize(fact["source"])

            if use_tfidf and self._idf:
                score = 0.0
                for qt in query_tokens:
                    tf = doc_counts.get(qt, 0) / max(len(doc_tokens), 1)
                    idf = self._idf.get(qt, 1.0)
                    score += tf * idf
                    if qt in source_tokens:
                        score += idf * 2
            else:
                score = 0
                text_lower = fact["text"].lower()
                source_lower = fact["source"].lower()
                for kw in query_tokens:
                    score += text_lower.count(kw) * 2
                    score += source_lower.count(kw) * 3

            scored.append((score, fact))

        scored.sort(key=lambda x: -x[0])
        top = scored[:top_k]
        top_positive = [(s, f) for s, f in top if s > 0]
        if not top_positive and scored:
            top_positive = scored[:min(2, len(scored))]

        results = []
        for score, fact in top_positive:
            if fact["type"] == "document" and "chunk_id" in fact:
                label = f"[Doc] {fact['source']} (partie {fact['chunk_id'] + 1})"
            else:
                label = f"[{fact['type']}] {fact['source']}"
            results.append(f"{label}: {fact['text'][:600]}")
        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Zéèêëàâäîïôûùç0-9']{2,}\b", text.lower())
        return [w for w in words if w not in AFRICAN_STOP_WORDS and not w.isdigit()]
