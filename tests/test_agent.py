from agent.models import LearnResponse
from agent.orchestrator import MicrosoftLearnAgent
from services.knowledge_generator import ExtractiveKnowledgeGenerator, select_grounding_chunks

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
        return {
            "title": "Azure Functions overview",
            "url": url,
            "headings": ["Overview"],
            "content": "## Overview\nAzure Functions runs event-driven code.",
            "image_url": "https://learn.microsoft.com/en-us/azure/media/functions-overview.png",
        }


def test_agent_runs_complete_grounded_flow_and_validates_sources():
    search, reader = FakeSearch(), FakeReader()
    agent = MicrosoftLearnAgent(search_tool=search, reader_tool=reader)

    response = agent.run("Explain Azure Functions and compare Consumption versus Premium plans", "Study Notes", 5)

    assert len(search.queries) > 1
    assert reader.urls == [LEARN_URL]
    assert [str(source.url) for source in response.sources] == [LEARN_URL]
    assert response.sources[0].title == "Azure Functions overview"
    assert str(response.sources[0].image_url) == "https://learn.microsoft.com/en-us/azure/media/functions-overview.png"
    assert response.definition == "Azure Functions runs event-driven code."
    assert response.summary == "Azure Functions runs event-driven code."
    assert response.notes[0].content == "Azure Functions runs event-driven code."
    assert str(response.notes[0].image_url) == "https://learn.microsoft.com/en-us/azure/media/functions-overview.png"
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
            "title": "What is Azure Functions?",
            "url": LEARN_URL,
            "description": "Azure Functions is a serverless compute service.",
            "score": 0.5,
        },
        {
            "title": "Azure Functions documentation",
            "url": "https://learn.microsoft.com/en-us/azure/azure-functions/",
            "description": "Azure Functions product documentation.",
            "score": 0.167,
        },
    ]

    sources = agent.select_sources(candidates, "Azure Functions", 5)

    assert len(sources) == 3
    assert sources[0]["url"] == "https://learn.microsoft.com/en-us/azure/azure-functions/"
    assert sources[1]["url"] == LEARN_URL


def test_specific_query_does_not_fill_sources_with_unrelated_results():
    agent = MicrosoftLearnAgent(search_tool=FakeSearch(), reader_tool=FakeReader())
    candidates = [
        {
            "title": "Create an autoscale scaling plan for Azure Virtual Desktop",
            "url": "https://learn.microsoft.com/en-us/azure/virtual-desktop/autoscale-create-assign-scaling-plan",
            "description": "Configure Azure Virtual Desktop autoscale scaling plans.",
            "score": 1.0,
        },
        {
            "title": "Azure App Service plans",
            "url": "https://learn.microsoft.com/en-us/azure/app-service/overview-hosting-plans",
            "description": "Choose an App Service plan.",
            "score": 0.5,
        },
    ]

    sources = agent.select_sources(candidates, "Azure Virtual Desktop autoscale scaling plans", 5)

    assert [source["title"] for source in sources] == ["Create an autoscale scaling plan for Azure Virtual Desktop"]


def test_specific_query_filters_incidental_mentions_in_retrieved_pages():
    agent = MicrosoftLearnAgent(search_tool=FakeSearch(), reader_tool=FakeReader())
    documents = [
        {
            "title": "Managed identities for Azure resources",
            "url": "https://learn.microsoft.com/en-us/entra/identity/managed-identities-azure-resources/overview",
            "headings": ["Managed identity types", "Use managed identities"],
            "content": "Managed identity " * 30,
        },
        {
            "title": "Develop Azure Functions locally",
            "url": "https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local",
            "headings": ["Develop locally", "Install Core Tools"],
            "content": ("Azure Functions local development. " * 200) + "Managed identity can be used.",
        },
    ]

    selected = agent.select_documents(documents, "Azure Functions managed identity")

    assert [document["title"] for document in selected] == ["Managed identities for Azure resources"]


def test_agent_uses_extractive_generator():
    agent = MicrosoftLearnAgent(search_tool=FakeSearch(), reader_tool=FakeReader())
    assert isinstance(agent.knowledge_generator, ExtractiveKnowledgeGenerator)


def test_grounding_chunks_are_interleaved_across_documents():
    documents = [
        {
            "title": title,
            "url": url,
            "headings": ["One", "Two"],
            "content": f"## One\nAzure Functions {title} first.\n## Two\nAzure Functions {title} second.",
        }
        for title, url in (("Overview", LEARN_URL), ("Hosting", SECOND_LEARN_URL))
    ]

    chunks = select_grounding_chunks("Azure Functions", documents, per_document=2)

    assert [chunk["title"] for chunk in chunks] == ["Overview", "Hosting", "Overview", "Hosting"]


def test_definition_prefers_introductory_explanation_over_later_configuration():
    document = {
        "title": "Create an autoscale scaling plan",
        "url": "https://learn.microsoft.com/en-us/azure/virtual-desktop/autoscale",
        "headings": ["Overview", "Ramp-down"],
        "content": (
            "## Overview\nAutoscale lets you scale session host virtual machines according to schedule.\n"
            "## Ramp-down\nThe capacity threshold is the percentage used by the scaling plan."
        ),
    }

    response = ExtractiveKnowledgeGenerator().generate(
        "Azure Virtual Desktop autoscale scaling plans", "Study Notes", [document]
    )

    assert response.definition == "Autoscale lets you scale session host virtual machines according to schedule."