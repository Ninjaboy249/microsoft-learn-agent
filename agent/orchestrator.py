"""Agentic retrieval and grounded generation workflow."""

from __future__ import annotations

import re
from typing import Any

from agent.models import LearnResponse, Source
from services.knowledge_generator import ExtractiveKnowledgeGenerator
from tools.content_processor import chunk_content, prepare_context
from tools.learn_reader import LearnReaderError, MicrosoftLearnReaderTool
from tools.learn_search import MicrosoftLearnSearchTool, is_microsoft_learn_url


class AgentError(RuntimeError):
    """A failure that can be displayed safely in the UI."""


class MicrosoftLearnAgent:
    def __init__(
        self,
        search_tool: Any | None = None,
        reader_tool: Any | None = None,
        knowledge_generator: Any | None = None,
    ) -> None:
        self.search_tool = search_tool or MicrosoftLearnSearchTool()
        self.reader_tool = reader_tool or MicrosoftLearnReaderTool()
        self.knowledge_generator = knowledge_generator or ExtractiveKnowledgeGenerator()

    def run(self, user_query: str, mode: str, max_sources: int = 5) -> LearnResponse:
        query = user_query.strip()
        if not query:
            raise AgentError("Enter a Microsoft technology topic to begin.")
        max_sources = max(1, min(max_sources, 8))

        intent = self.understand_query(query)
        search_queries = self.generate_search_queries(intent)
        candidates = self.search_sources(search_queries, max_sources)
        sources = self.select_sources(candidates, intent["topic"], max_sources)
        if not sources:
            raise AgentError("No relevant Microsoft Learn documentation was found for this topic.")

        documents = self.retrieve_documents(sources)
        if not intent["compound"]:
            documents = self.select_documents(documents, intent["topic"])
        if not documents:
            raise AgentError("Microsoft Learn results were found, but the pages could not be read.")
        context = self.build_context(documents)
        if not context:
            raise AgentError("The retrieved pages did not contain enough readable documentation.")

        response = self.generate_response(query, mode, documents)
        return self.validate_sources(response, documents)

    def understand_query(self, query: str) -> dict:
        """Classify retrieval breadth and identify explicit comparison subjects."""
        normalized = re.sub(r"\s+", " ", query).strip()
        comparison = bool(re.search(r"\b(compare|comparison|versus|vs\.?|difference|between)\b", normalized, re.I))
        compound = comparison or bool(re.search(r"\b(and|when|how|why)\b", normalized, re.I))
        cleaned = re.sub(r"^(explain|describe|what is|tell me about)\s+", "", normalized, flags=re.I)
        parts = [part.strip(" .?") for part in re.split(r"\b(?:versus|vs\.?|and|compared? to)\b", cleaned, flags=re.I)]
        concepts = [part for part in parts if len(part.split()) <= 10 and len(part) > 2]
        return {"original": normalized, "topic": cleaned, "comparison": comparison, "compound": compound, "concepts": concepts}

    def generate_search_queries(self, intent: dict) -> list[str]:
        """Choose one broad search for simple intent and focused searches for complex intent."""
        queries = [intent["topic"]]
        if intent["compound"]:
            queries.extend(intent["concepts"][:3])
        if intent["comparison"] and len(intent["concepts"]) > 1:
            queries.append(f"{intent['concepts'][0]} {intent['concepts'][-1]} comparison")
        return list(dict.fromkeys(query for query in queries if query))[:4]

    def search_sources(self, queries: list[str], max_sources: int) -> list[dict]:
        candidates: list[dict] = []
        candidate_limit = min(max(max_sources * 2, 8), 10)
        for query in queries:
            try:
                for result in self.search_tool.search(query, candidate_limit):
                    candidate = dict(result)
                    candidate["matched_query"] = query
                    candidates.append(candidate)
            except Exception:
                continue
        return candidates

    def select_sources(self, candidates: list[dict], query: str, max_sources: int) -> list[dict]:
        query_words = self._search_words(query)
        query_terms = set(query_words)
        normalized_query = " ".join(query_words)
        unique: dict[str, dict] = {}
        for candidate in candidates:
            url = str(candidate.get("url", "")).split("#", 1)[0]
            if not is_microsoft_learn_url(url):
                continue
            title = str(candidate.get("title", ""))
            searchable = f"{title} {candidate.get('description', '')}".casefold()
            overlap = sum(term in searchable for term in query_terms)
            title_words = self._search_words(title)
            title_terms = set(title_words)
            normalized_title = " ".join(title_words)
            normalized_title = re.sub(r"\s+(documentation|docs)(\s+learn)?$", "", normalized_title)
            landing_bonus = 5 if normalized_title == normalized_query else 0
            title_coverage = len(query_terms & title_terms) / max(len(query_terms), 1)
            exact_phrase_bonus = 2 if normalized_query in " ".join(title_words) else 0
            overview_bonus = 2 if title_coverage == 1 and re.search(r"\b(overview|documentation|introduction|what is)\b", title, re.I) else 0
            extra_title_terms = max(len(title_terms - query_terms), 0)
            specificity_penalty = min(extra_title_terms * 0.2, 2.0) if len(query_terms) <= 3 else 0
            ranked = dict(
                candidate,
                url=url,
                rank_score=(
                    float(candidate.get("score", 0))
                    + overlap
                    + title_coverage * 4
                    + exact_phrase_bonus
                    + overview_bonus
                    + landing_bonus
                    - specificity_penalty
                ),
                canonical_for_query=bool(landing_bonus),
                query_coverage=len(query_terms & set(self._search_words(searchable))) / max(len(query_terms), 1),
            )
            if url not in unique or ranked["rank_score"] > unique[url]["rank_score"]:
                unique[url] = ranked
        ranked_sources = sorted(unique.values(), key=lambda item: item["rank_score"], reverse=True)
        if not ranked_sources:
            return []
        minimum_ratio = 0.5 if len(query_terms) <= 3 else 0.65
        minimum_score = ranked_sources[0]["rank_score"] * minimum_ratio
        relevant = [
            source for source in ranked_sources
            if source["rank_score"] >= minimum_score and source["query_coverage"] >= 0.5
        ]
        return relevant[:max_sources]

    @staticmethod
    def _search_words(text: str) -> list[str]:
        words: list[str] = []
        for word in re.findall(r"[a-z0-9]+", text.casefold()):
            if len(word) > 4 and word.endswith("ies"):
                word = f"{word[:-3]}y"
            elif len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
                word = word[:-1]
            words.append(word)
        return words

    def retrieve_documents(self, sources: list[dict]) -> list[dict]:
        documents: list[dict] = []
        for source in sources:
            try:
                documents.append(self.reader_tool.get_page(source["url"]))
            except (LearnReaderError, ValueError):
                continue
        return documents

    def select_documents(self, documents: list[dict], query: str) -> list[dict]:
        """Drop pages whose extracted content only mentions a specific query incidentally."""
        query_words = self._search_words(query)
        query_terms = set(query_words)
        if len(query_terms) <= 3 or len(documents) <= 1:
            return documents

        phrases = [" ".join(query_words[index:index + 2]) for index in range(len(query_words) - 1)]
        focus_phrases = phrases[-2:]
        scored: list[tuple[float, dict]] = []
        for document in documents:
            title_terms = set(self._search_words(str(document.get("title", ""))))
            heading_text = " ".join(str(heading) for heading in document.get("headings", []))
            heading_terms = set(self._search_words(heading_text))
            content = str(document.get("content", "")).casefold()
            content_terms = set(self._search_words(content))
            title_coverage = len(query_terms & title_terms) / len(query_terms)
            heading_coverage = len(query_terms & heading_terms) / len(query_terms)
            content_coverage = len(query_terms & content_terms) / len(query_terms)
            phrase_hits = sum(content.count(phrase) for phrase in focus_phrases)
            phrase_density = min(phrase_hits * 10_000 / max(len(content), 1), 3.0)
            score = title_coverage * 6 + heading_coverage * 4 + content_coverage + phrase_density
            scored.append((score, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        minimum_score = scored[0][0] * 0.6
        return [document for score, document in scored if score >= minimum_score]

    def build_context(self, documents: list[dict]) -> str:
        chunks: list[dict] = []
        for document in documents:
            chunks.extend(chunk_content(document)[:3])
        return prepare_context(chunks)

    def generate_response(self, query: str, mode: str, documents: list[dict]) -> LearnResponse:
        return self.knowledge_generator.generate(query, mode, documents)

    def validate_sources(self, response: LearnResponse, documents: list[dict]) -> LearnResponse:
        """Remove malformed, external, duplicate, or unretrieved citations."""
        retrieved = {document["url"].split("#", 1)[0]: document for document in documents}
        validated: list[Source] = []
        seen: set[str] = set()
        for source in response.sources:
            url = str(source.url).split("#", 1)[0]
            if url in retrieved and url not in seen and is_microsoft_learn_url(url):
                validated.append(Source(title=retrieved[url]["title"], url=url))
                seen.add(url)
        return response.model_copy(update={"sources": validated})