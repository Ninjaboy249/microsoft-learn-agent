"""Clean, chunk, and assemble retrieved Microsoft Learn documentation."""

from __future__ import annotations

import re

MAX_CONTEXT_CHARACTERS = 100_000

IGNORED_LINE_PATTERNS = (
    re.compile(r"access to this page requires authorization", re.I),
    re.compile(r"you can try (?:signing in or )?changing directories", re.I),
)


def clean_content(content: str) -> str:
    """Normalize extracted text while preserving paragraphs and headings."""
    if not content:
        return ""
    content = content.replace("\xa0", " ").replace("\r\n", "\n")
    content = re.sub(r"[ \t]+", " ", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    lines: list[str] = []
    previous = ""
    for line in (part.strip() for part in content.splitlines()):
        if any(pattern.search(line) for pattern in IGNORED_LINE_PATTERNS):
            if lines and lines[-1].casefold() == "note":
                lines.pop()
            continue
        if line and line != previous:
            lines.append(line)
            previous = line
    return "\n".join(lines).strip()


def chunk_content(document: dict, chunk_size: int = 5_000) -> list[dict]:
    """Split a document near section and paragraph boundaries."""
    content = clean_content(str(document.get("content", "")))
    if not content:
        return []
    title = str(document.get("title", "Microsoft Learn"))
    url = str(document.get("url", ""))
    chunks: list[dict] = []
    section = title
    buffer: list[str] = []
    buffer_length = 0
    code_buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, buffer_length
        if buffer:
            chunks.append({"title": title, "section": section, "content": "\n".join(buffer), "url": url})
            buffer = []
            buffer_length = 0

    def append_part(part: str) -> None:
        nonlocal buffer_length
        if buffer and buffer_length + len(part) + 1 > chunk_size:
            flush()
        buffer.append(part)
        buffer_length += len(part) + 1

    for line in content.splitlines():
        if line == "```":
            code_buffer.append(line)
            if len(code_buffer) > 1:
                for part in _split_code_block("\n".join(code_buffer), chunk_size):
                    append_part(part)
                code_buffer = []
            continue
        if code_buffer:
            code_buffer.append(line)
            continue
        if line.startswith("## "):
            flush()
            section = line[3:].strip() or title
            continue
        for part in _split_long_text(line, chunk_size):
            append_part(part)
    if code_buffer:
        for part in _split_code_block("\n".join([*code_buffer, "```"]), chunk_size):
            append_part(part)
    flush()
    return chunks


def _split_long_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    return [text[index : index + limit] for index in range(0, len(text), limit)]


def _split_code_block(block: str, limit: int) -> list[str]:
    inner = block.removeprefix("```\n").removesuffix("\n```")
    content_limit = max(limit - 8, 1)
    return [f"```\n{part}\n```" for part in _split_long_text(inner, content_limit)]


def prepare_context(chunks: list[dict], max_characters: int = MAX_CONTEXT_CHARACTERS) -> str:
    """Create bounded context with explicit source metadata and trust markers."""
    sections: list[str] = []
    used = 0
    for chunk in chunks:
        block = (
            f"SOURCE TITLE: {chunk['title']}\nSOURCE URL: {chunk['url']}\n"
            f"SECTION: {chunk['section']}\nUNTRUSTED DOCUMENTATION DATA:\n{chunk['content']}"
        )
        remaining = max_characters - used
        if remaining <= 0:
            break
        block = block[:remaining]
        sections.append(block)
        used += len(block) + 6
    return "\n\n---\n\n".join(sections)