import httpx
import pytest

from tools.learn_reader import LearnReaderError, MicrosoftLearnReaderTool


def test_reader_rejects_external_url_without_request():
    with pytest.raises(ValueError, match="Only https://learn.microsoft.com"):
        MicrosoftLearnReaderTool().get_page("https://example.org/documentation")


def test_reader_extracts_documentation_and_removes_ui():
    html = """
    <html><head><title>Fallback</title><script>bad()</script></head><body>
      <header>Global navigation</header><main>
        <h1>Azure Functions overview</h1><p>Run event-driven code.</p>
        <h2>Hosting</h2><ul><li>Consumption plan</li><li>Premium plan</li></ul>
        <img src="/en-us/azure/media/functions-overview.svg" alt="Azure Functions architecture">
        <img src="https://example.org/untrusted.png" alt="External image">
        <pre>func start\nfunc azure functionapp publish example-app</pre><nav>Related UI</nav>
      </main><footer>Footer</footer>
    </body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, text=html)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    page = MicrosoftLearnReaderTool(client).get_page("https://learn.microsoft.com/en-us/test/reader-extraction")

    assert page["title"] == "Azure Functions overview"
    assert page["headings"] == ["Azure Functions overview", "Hosting"]
    assert page["image_url"] == "https://learn.microsoft.com/en-us/azure/media/functions-overview.svg"
    assert page["section_images"]["Hosting"] == page["image_url"]
    assert "Consumption plan" in page["content"]
    assert "```\nfunc start\nfunc azure functionapp publish example-app\n```" in page["content"]
    assert "Global navigation" not in page["content"]
    assert "bad()" not in page["content"]


def test_reader_wraps_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(LearnReaderError, match="Unable to retrieve"):
        MicrosoftLearnReaderTool(client).get_page("https://learn.microsoft.com/en-us/test/reader-timeout")