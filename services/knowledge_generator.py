"""Generate structured notes directly from retrieved Microsoft Learn text."""

from __future__ import annotations

import re
from collections import Counter

from agent.models import LearnResponse, Note, Source
from tools.content_processor import chunk_content, clean_content

STOP_WORDS = {
    "about", "after", "also", "and", "are", "been", "before", "can", "for",
    "describe", "does", "explain", "from", "have", "how", "into", "is", "more",
    "not", "only", "that", "the", "their", "this", "through", "use", "using",
    "what", "when", "where", "which", "with", "your",
}

GENERIC_CONCEPTS = {
    "concept", "documentation", "get started", "how-to guide", "introduction",
    "overview", "quickstart", "reference", "sample", "tutorial",
}

GENERIC_NOTE_HEADINGS = {"concept", "how-to guide", "quickstart", "reference", "sample", "tutorial"}

MODE_LIMITS = {
    "Quick Summary": (3, 3, 0),
    "Study Notes": (7, 10, 12),
    "Beginner Explanation": (5, 7, 8),
    "Deep Dive": (10, 14, 18),
    "Interview Preparation": (7, 10, 12),
    "Step-by-Step Guide": (7, 12, 16),
    "Quiz": (6, 10, 10),
}

def select_grounding_chunks(query: str, documents: list[dict], per_document: int = 8) -> list[dict]:
    """Select relevant chunks from every retrieved document for balanced grounding."""
    query_terms = set(re.findall(r"[a-z0-9][a-z0-9+#.-]*", query.casefold())) - STOP_WORDS
    ranked_documents: list[list[dict]] = []
    for document in documents:
        chunks = chunk_content(document)
        ranked = sorted(
            enumerate(chunks),
            key=lambda item: (
                len(query_terms & set(re.findall(
                    r"[a-z0-9][a-z0-9+#.-]*",
                    f"{item[1]['section']} {item[1]['content']}".casefold(),
                ))),
                -item[0],
            ),
            reverse=True,
        )[:per_document]
        ranked_documents.append([chunk for _, chunk in ranked])
    selected: list[dict] = []
    for rank in range(per_document):
        for chunks in ranked_documents:
            if rank < len(chunks):
                selected.append(chunks[rank])
    return selected


class ExtractiveKnowledgeGenerator:
    """Select source text without inventing or paraphrasing documentation."""

    def generate(self, query: str, mode: str, documents: list[dict]) -> LearnResponse:
        summary_limit, concept_limit, note_limit = MODE_LIMITS.get(mode, MODE_LIMITS["Study Notes"])
        chunks = select_grounding_chunks(query, documents, per_document=12)
        sentences = self._ranked_sentences(query, chunks)
        definition = self._select_definition(query, chunks)
        summary_sentences = ([definition] if definition else []) + [
            sentence for _, sentence in sentences if sentence != definition
        ][:max(summary_limit - bool(definition), 0)]
        summary = " ".join(summary_sentences) or "No concise text could be extracted from the retrieved documentation."

        notes = self._build_notes(query, chunks, note_limit)
        examples = self._extract_examples(chunks, limit=4)
        concepts = self._extract_concepts(query, documents, chunks, concept_limit)
        sources = [
            Source(title=document["title"], url=document["url"], image_url=document.get("image_url"))
            for document in documents
        ]

        return LearnResponse(
            topic=query,
            definition=definition or summary_sentences[0] if summary_sentences else summary,
            summary=summary,
            key_concepts=concepts,
            notes=notes,
            examples=examples,
            sources=sources,
        )

    def _select_definition(self, query: str, chunks: list[dict]) -> str:
        query_terms = self._terms(query)
        candidates: list[tuple[float, str]] = []
        definition_pattern = re.compile(
            r"\b(?:is|are)\s+(?:an?|the)\b|\b(?:refers to|lets you|allows you to|enables you to|provides)\b",
            re.I,
        )
        for chunk_index, chunk in enumerate(chunks):
            section = chunk["section"].casefold()
            section_bonus = 4 if re.search(r"overview|introduction|what is|documentation", section) else 0
            for sentence_index, sentence in enumerate(self._sentences(chunk["content"])):
                overlap = len(query_terms & self._terms(sentence))
                if overlap == 0 or not definition_pattern.search(sentence) or len(sentence) > 700:
                    continue
                coverage = overlap / max(len(query_terms), 1)
                score = overlap * 3 + coverage * 4 + section_bonus - chunk_index * 0.3 - sentence_index * 0.05
                candidates.append((score, sentence))
        return max(candidates, default=(0.0, ""), key=lambda item: item[0])[1]

    def _ranked_sentences(self, query: str, chunks: list[dict]) -> list[tuple[float, str]]:
        query_terms = self._terms(query)
        ranked: list[tuple[float, str]] = []
        seen: set[str] = set()
        for chunk_index, chunk in enumerate(chunks):
            for sentence in self._sentences(chunk["content"]):
                normalized = sentence.casefold()
                if normalized in seen or len(sentence) < 35:
                    continue
                seen.add(normalized)
                sentence_terms = self._terms(sentence)
                overlap = len(query_terms & sentence_terms)
                score = overlap * 4 + min(len(sentence_terms), 30) / 30 - chunk_index * 0.02
                ranked.append((score, sentence))
        return sorted(ranked, key=lambda item: item[0], reverse=True)

    def _build_notes(self, query: str, chunks: list[dict], limit: int) -> list[Note]:
        if limit == 0:
            return []
        query_terms = self._terms(query)
        ranked_chunks = sorted(
            chunks,
            key=lambda chunk: (
                len(query_terms & self._terms(chunk["section"])) * 3
                + len(query_terms & self._terms(chunk["content"])),
                min(len(re.sub(r"```.*?```", "", chunk["content"], flags=re.DOTALL)), 3_000),
            ),
            reverse=True,
        )
        notes: list[Note] = []
        seen_sections: set[str] = set()
        for chunk in ranked_chunks:
            section = chunk["section"].strip()
            if section.casefold() in seen_sections or section.casefold() in GENERIC_NOTE_HEADINGS:
                continue
            content = clean_content(re.sub(r"```.*?```", "", chunk["content"], flags=re.DOTALL))
            if not content:
                continue
            seen_sections.add(section.casefold())
            notes.append(Note(heading=section, content=content[:3500]))
            if len(notes) == limit:
                break
        return notes

    def _extract_concepts(self, query: str, documents: list[dict], chunks: list[dict], limit: int) -> list[str]:
        candidates: list[str] = []
        for document in documents:
            candidates.extend(document.get("headings", []))
        candidates.extend(chunk["section"] for chunk in chunks)
        query_terms = self._terms(query)
        counts = Counter(term for chunk in chunks for term in self._terms(chunk["content"]) if len(term) > 3)
        candidates.extend(term.title() for term, _ in counts.most_common(limit * 2) if term not in query_terms)

        concepts: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            value = re.sub(r"\s+", " ", str(candidate)).strip(" .:#")
            if not value or value.casefold() in seen or value.casefold() in GENERIC_CONCEPTS:
                continue
            seen.add(value.casefold())
            concepts.append(value)
            if len(concepts) == limit:
                break
        return concepts

    @staticmethod
    def _extract_examples(chunks: list[dict], limit: int) -> list[str]:
        examples: list[str] = []
        for chunk in chunks:
            for block in re.findall(r"```\s*(.*?)\s*```", chunk["content"], re.DOTALL):
                value = clean_content(block)
                if value and value not in examples:
                    examples.append(value[:1000])
                    if len(examples) == limit:
                        return examples
        return examples

    @staticmethod
    def _sentences(content: str) -> list[str]:
        plain = re.sub(r"```.*?```", " ", content, flags=re.DOTALL)
        sentences = [part.strip(" -\n") for part in re.split(r"(?<=[.!?])\s+|\n+", plain) if part.strip()]
        return [sentence for sentence in sentences if sentence.endswith((".", "!", "?"))]

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {term for term in re.findall(r"[a-z0-9][a-z0-9+#.-]*", text.casefold()) if term not in STOP_WORDS}