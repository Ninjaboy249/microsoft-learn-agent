import json
from types import SimpleNamespace

from agent.models import LearnResponse
from agent.orchestrator import MicrosoftLearnAgent
from services.knowledge_generator import AzureOpenAIKnowledgeGenerator

LEARN_URL = "https://learn.microsoft.com/en-us/azure/azure-functions/functions-overview"
SECOND_LEARN_URL = "https://learn.microsoft.com/en-us/azure/azure-functions/functions-scale"
EXTERNAL_URL = "https://example.org/fake"


class FakeSearch:
    def __init__(self):
        self.queries = []

    def search(self, query, max_results):
        self.queries.append(query)
        return [
            {"title": "Azure Functions overview", "url": LEARN_URL, "description": "Functions hosting Consumption Premium", "score": 1.0},
            {"title": "Fake", "url": EXTERNAL_URL, "description": "External", "score": 5.0},
        ]


class FakeReader:
    def __init__(self):
        self.urls = []

    def get_page(self, url):
        self.urls.append(url)
        return {"title": "Azure Functions overview", "url": url, "headings": ["Overview"], "content": "## Overview\nAzure Functions runs event-driven code."}


def test_agent_runs_complete_grounded_flow_and_validates_sources():
    search, reader = FakeSearch(), FakeReader()
    agent = MicrosoftLearnAgent(search_tool=search, reader_tool=reader)

    response = agent.run("Explain Azure Functions and compare Consumption versus Premium plans", "Study Notes", 5)

    assert len(search.queries) > 1
    assert reader.urls == [LEARN_URL]
    assert [str(source.url) for source in response.sources] == [LEARN_URL]
    assert response.sources[0].title == "Azure Functions overview"
    assert response.definition == "Azure Functions runs event-driven code."
    assert response.summary == "Azure Functions runs event-driven code."
    assert response.notes[0].content == "Azure Functions runs event-driven code."
    assert LearnResponse.model_validate(response.model_dump()) == response


def test_simple_query_uses_one_search():
    agent = MicrosoftLearnAgent(search_tool=FakeSearch(), reader_tool=FakeReader())
    intent = agent.understand_query("What is Azure Blob Storage?")
    assert agent.generate_search_queries(intent) == ["Azure Blob Storage?"]


def test_broad_product_query_prioritizes_landing_page_and_keeps_relevant_sources():
    agent = MicrosoftLearnAgent(search_tool=FakeSearch(), reader_tool=FakeReader())
    candidates = [
        {
            "title": "Develop Azure Functions Locally by using Core Tools",
            "url": "https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local",
            "description": "Develop and test functions locally.",
            "score": 1.0,
        },
        {
            "title": "Azure Functions documentation",
            "url": "https://learn.microsoft.com/en-us/azure/azure-functions/",
            "description": "Azure Functions product documentation.",
            "score": 0.167,
        },
    ]

    sources = agent.select_sources(candidates, "Azure Functions", 5)

    assert len(sources) == 2
    assert sources[0]["url"] == "https://learn.microsoft.com/en-us/azure/azure-functions/"


def test_agent_initializes_without_api_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    agent = MicrosoftLearnAgent(search_tool=FakeSearch(), reader_tool=FakeReader())
    assert agent.knowledge_generator.__class__.__name__ == "ExtractiveKnowledgeGenerator"


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_azure_generator_uses_retrieved_content_and_preserves_sources():
    completions = FakeCompletions(json.dumps({
        "topic": "ignored",
        "summary": "A generated, grounded explanation.",
        "key_concepts": ["Events"],
        "notes": [{"heading": "Overview", "content": "Functions run event-driven code."}],
        "examples": [],
    }))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    generator = AzureOpenAIKnowledgeGenerator("key", "endpoint", "deployment", client=client)
    document = FakeReader().get_page(LEARN_URL)
    hosting_document = {
        "title": "Azure Functions hosting options",
        "url": SECOND_LEARN_URL,
        "headings": ["Hosting"],
        "content": "## Hosting\nAzure Functions provides multiple hosting options.",
    }

    response = generator.generate("Azure Functions", "Beginner Explanation", [document, hosting_document])

    assert response.summary == "A generated, grounded explanation."
    assert response.definition == "A generated, grounded explanation."
    assert response.topic == "Azure Functions"
    assert [str(source.url) for source in response.sources] == [LEARN_URL, SECOND_LEARN_URL]
    request_context = completions.requests[0]["messages"][1]["content"]
    assert "Azure Functions runs event-driven code." in request_context
    assert "Azure Functions provides multiple hosting options." in request_context


def test_azure_generator_falls_back_when_model_output_is_invalid():
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions("not json")))
    generator = AzureOpenAIKnowledgeGenerator("key", "endpoint", "deployment", client=client)

    response = generator.generate("Azure Functions", "Study Notes", [FakeReader().get_page(LEARN_URL)])

    assert response.summary == "Azure Functions runs event-driven code."