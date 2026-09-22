from tools.content_processor import chunk_content, clean_content, prepare_context


def test_clean_content_removes_duplicate_lines_and_whitespace():
    assert clean_content(" First   line \nFirst line\n\n\nSecond\tline ") == "First line\nSecond line"
    assert clean_content("") == ""


def test_clean_content_removes_authorization_boilerplate():
    content = (
        "Useful documentation.\n"
        "Note\nAccess to this page requires authorization. You can try signing in or changing directories.\n"
        "More useful documentation."
    )

    assert clean_content(content) == "Useful documentation.\nMore useful documentation."


def test_clean_content_preserves_code_lines_and_indentation():
    content = "Before.\n```\nfunc start\n  --verbose\n```\nAfter."

    assert clean_content(content) == content


def test_chunking_preserves_source_and_sections():
    document = {
        "title": "Functions",
        "url": "https://learn.microsoft.com/en-us/functions",
        "content": "## Overview\n" + ("Azure Functions content. " * 20) + "\n## Hosting\nPremium details.",
    }
    chunks = chunk_content(document, chunk_size=120)

    assert len(chunks) > 2
    assert {chunk["section"] for chunk in chunks} >= {"Overview", "Hosting"}
    assert all(chunk["url"] == document["url"] for chunk in chunks)
    assert all(len(chunk["content"]) <= 120 for chunk in chunks)


def test_chunking_keeps_long_code_inside_balanced_fences():
    document = {
        "title": "Functions",
        "url": "https://learn.microsoft.com/en-us/functions",
        "content": "## Overview\nUseful prose.\n```\n" + ("const azureFunction = true; " * 20) + "\n```\nMore prose.",
    }

    chunks = chunk_content(document, chunk_size=120)

    assert all(chunk["content"].count("```") % 2 == 0 for chunk in chunks)
    assert all(len(chunk["content"]) <= 120 for chunk in chunks)


def test_prepare_context_enforces_maximum_size():
    chunks = [{"title": "Title", "section": "Section", "content": "x" * 500, "url": "https://learn.microsoft.com/test"}]
    context = prepare_context(chunks, max_characters=100)
    assert len(context) <= 100
    assert "SOURCE TITLE" in context
    assert chunk_content({"title": "Empty", "url": "https://learn.microsoft.com/", "content": ""}) == []