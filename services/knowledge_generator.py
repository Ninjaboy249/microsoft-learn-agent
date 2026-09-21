"""Generate structured notes directly from retrieved Microsoft Learn text."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

from agent.models import LearnResponse, Note, Source
from tools.content_processor import chunk_content, clean_content, prepare_context

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
    "Study Notes": (5, 6, 6),
    "Beginner Explanation": (4, 5, 4),
    "Deep Dive": (7, 8, 8),
    "Interview Preparation": (5, 7, 6),
    "Step-by-Step Guide": (5, 8, 8),
    "Quiz": (5, 8, 6),
}

AZURE_SETTING_NAMES = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
)


def select_grounding_chunks(query: str, documents: list[dict], per_document: int = 4) -> list[dict]:
    """Select relevant chunks from every retrieved document for balanced grounding."""
    query_terms = set(re.findall(r"[a-z0-9][a-z0-9+#.-]*", query.casefold())) - STOP_WORDS
    selected: list[dict] = []
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
        selected.extend(chunk for _, chunk in sorted(ranked, key=lambda item: item[0]))
    return selected


class AzureOpenAIKnowledgeGenerator:
    """Generate a grounded response from retrieved content using Azure OpenAI."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment: str,
        api_version: str = "2024-10-21",
        client: Any | None = None,
        fallback: Any | None = None,
    ) -> None:
        if client is None:
            from openai import AzureOpenAI

            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
            )
        self.client = client
        self.deployment = deployment
        self.fallback = fallback or ExtractiveKnowledgeGenerator()

    def generate(self, query: str, mode: str, documents: list[dict]) -> LearnResponse:
        chunks = select_grounding_chunks(query, documents)
        context = prepare_context(chunks)
        sources = [Source(title=document["title"], url=document["url"]) for document in documents]
        try:
            completion = self.client.chat.completions.create(
                model=self.deployment,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You create accurate learning material using only the supplied Microsoft Learn "
                            "content. Do not add unsupported facts. Return JSON with string fields topic and "
                            "definition and summary, string arrays key_concepts and examples, and a notes array "
                            "containing objects with heading and content. Definition must directly state what the "
                            "topic is. Synthesize the relevant facts across all supplied sources, merge duplicate "
                            "information, and preserve important capabilities, limitations, warnings, prerequisites, "
                            "and procedures. Organize notes with clear descriptive headings appropriate to the "
                            "requested mode. Do not include sources in the JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Topic: {query}\nMode: {mode}\n\nMicrosoft Learn content:\n{context}",
                    },
                ],
            )
            content = completion.choices[0].message.content
            payload = json.loads(content or "{}")
            payload["topic"] = query
            payload["definition"] = str(payload.get("definition") or self._first_sentence(payload.get("summary", "")))
            payload["sources"] = [source.model_dump(mode="json") for source in sources]
            return LearnResponse.model_validate(payload)
        except Exception:
            return self.fallback.generate(query, mode, documents)

    @staticmethod
    def _first_sentence(text: str) -> str:
        match = re.search(r"^.*?[.!?](?:\s|$)", str(text).strip())
        return match.group(0).strip() if match else str(text).strip()


def create_knowledge_generator(settings: dict[str, str] | None = None) -> Any:
    """Use Azure OpenAI only when every required setting is available."""
    configured = {name: str(value).strip() for name, value in (settings or os.environ).items()}
    if not all(configured.get(name) for name in AZURE_SETTING_NAMES):
        return ExtractiveKnowledgeGenerator()
    return AzureOpenAIKnowledgeGenerator(
        api_key=configured["AZURE_OPENAI_API_KEY"],
        endpoint=configured["AZURE_OPENAI_ENDPOINT"],
        deployment=configured["AZURE_OPENAI_DEPLOYMENT"],
        api_version=configured.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


class ExtractiveKnowledgeGenerator:
    """Select source text without inventing or paraphrasing documentation."""

    def generate(self, query: str, mode: str, documents: list[dict]) -> LearnResponse:
        summary_limit, concept_limit, note_limit = MODE_LIMITS.get(mode, MODE_LIMITS["Study Notes"])
        chunks = [chunk for document in documents for chunk in chunk_content(document)]
        sentences = self._ranked_sentences(query, chunks)
        definition = self._select_definition(query, sentences)
        summary_sentences = ([definition] if definition else []) + [
            sentence for _, sentence in sentences if sentence != definition
        ][:max(summary_limit - bool(definition), 0)]
        summary = " ".join(summary_sentences) or "No concise text could be extracted from the retrieved documentation."

        notes = self._build_notes(query, chunks, note_limit)
        examples = self._extract_examples(chunks, limit=4)
        concepts = self._extract_concepts(query, documents, chunks, concept_limit)
        sources = [Source(title=document["title"], url=document["url"]) for document in documents]

        return LearnResponse(
            topic=query,
            definition=definition or summary_sentences[0] if summary_sentences else summary,
            summary=summary,
            key_concepts=concepts,
            notes=notes,
            examples=examples,
            sources=sources,
        )

    def _select_definition(self, query: str, sentences: list[tuple[float, str]]) -> str:
        query_terms = self._terms(query)
        candidates: list[tuple[float, str]] = []
        for score, sentence in sentences:
            overlap = len(query_terms & self._terms(sentence))
            if overlap == 0 or not re.search(r"\b(?:is|are|refers to|provides)\b", sentence, re.I):
                continue
            coverage = overlap / max(len(query_terms), 1)
            candidates.append((score + coverage * 8, sentence))
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
                len(query_terms & self._terms(f"{chunk['section']} {chunk['content']}")),
                -len(chunk["content"]),
            ),
            reverse=True,
        )
        notes: list[Note] = []
        seen_sections: set[str] = set()
        for chunk in ranked_chunks:
            section = chunk["section"].strip()
            if section.casefold() in seen_sections or section.casefold() in GENERIC_NOTE_HEADINGS:
                continue
            content = clean_content(chunk["content"])
            if not content:
                continue
            seen_sections.add(section.casefold())
            notes.append(Note(heading=section, content=content[:1800]))
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