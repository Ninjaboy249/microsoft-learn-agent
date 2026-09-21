import httpx

from tools.learn_search import MicrosoftLearnSearchTool, is_microsoft_learn_url


def test_url_validation_accepts_only_secure_exact_host():
    assert is_microsoft_learn_url("https://learn.microsoft.com/en-us/azure/functions/")
    assert not is_microsoft_learn_url("http://learn.microsoft.com/en-us/azure/")
    assert not is_microsoft_learn_url("https://learn.microsoft.com.example.org/fake")
    assert not is_microsoft_learn_url("https://example.org/")


def test_search_parses_filters_and_deduplicates_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "results": [
                    {"title": "Functions", "url": "https://learn.microsoft.com/en-us/azure/azure-functions/functions-overview", "description": "<b>Overview</b>"},
                    {"title": "Duplicate", "url": "https://learn.microsoft.com/en-us/azure/azure-functions/functions-overview", "description": "Same"},
                    {"title": "External", "url": "https://example.org/functions", "description": "No"},
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    results = MicrosoftLearnSearchTool(client).search("unique functions test", 5)

    assert results == [
        {
            "title": "Functions",
            "url": "https://learn.microsoft.com/en-us/azure/azure-functions/functions-overview",
            "description": "Overview",
            "score": 1.0,
        }
    ]


def test_search_handles_empty_results_and_failures():
    empty = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request, json={"results": []})))
    assert MicrosoftLearnSearchTool(empty).search("unique empty test") == []

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    failed = httpx.Client(transport=httpx.MockTransport(fail))
    assert MicrosoftLearnSearchTool(failed).search("unique timeout test") == []
    assert MicrosoftLearnSearchTool(empty).search("") == []