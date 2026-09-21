"""Agentic retrieval and grounded generation workflow."""

from __future__ import annotations

import re
from typing import Any

from agent.models import LearnResponse, Source
from services.knowledge_generator import create_knowledge_generator
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
        self.knowledge_generator = knowledge_generator or create_knowledge_generator()

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
        query_words = re.findall(r"[a-z0-9]+", query.casefold())
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
            normalized_title = " ".join(re.findall(r"[a-z0-9]+", title.casefold()))
            normalized_title = re.sub(r"\s+(documentation|docs)(\s+learn)?$", "", normalized_title)
            landing_bonus = 5 if normalized_title == normalized_query else 0
            ranked = dict(
                candidate,
                url=url,
                rank_score=float(candidate.get("score", 0)) + overlap + landing_bonus,
                canonical_for_query=bool(landing_bonus),
            )
            if url not in unique or ranked["rank_score"] > unique[url]["rank_score"]:
                unique[url] = ranked
        ranked_sources = sorted(unique.values(), key=lambda item: item["rank_score"], reverse=True)
        return ranked_sources[:max_sources]

    def retrieve_documents(self, sources: list[dict]) -> list[dict]:
        documents: list[dict] = []
        for source in sources:
            try:
                documents.append(self.reader_tool.get_page(source["url"]))
            except (LearnReaderError, ValueError):
                continue
        return documents

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