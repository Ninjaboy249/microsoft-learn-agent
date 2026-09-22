# Microsoft Learn Knowledge Generator

A keyless Streamlit application that searches and extracts documentation from `https://learn.microsoft.com/`, then creates a grounded answer with its built-in extractive generator.

## Features

- Searches the public Microsoft Learn search endpoint
- Accepts only exact HTTPS `learn.microsoft.com` URLs
- Downloads pages with `httpx`
- Extracts headings, paragraphs, lists, and code blocks with Beautiful Soup
- Ranks extracted sentences against the user's topic
- Produces summaries, key concepts, section notes, examples, and references
- Presents answers in a product-aware reading layout with trusted Microsoft Learn diagrams when available
- Requires no API key or cloud AI permission
- Supports simple and compound searches
- Caches searches and pages in process memory
- Stores recent topics only in the Streamlit session

## Architecture

```text
User
  |
  v
Streamlit UI
  |
  v
Query analyzer -> Microsoft Learn search
                         |
                         v
                  Secure page reader
                         |
                         v
               Clean and chunk content
                         |
                         v
             Extractive ranking and notes
                         |
                         v
          Structured response + exact URLs
```

## Project Structure

```text
microsoft-learn-agent/
|-- app.py
|-- agent/
|   |-- models.py
|   `-- orchestrator.py
|-- tools/
|   |-- learn_search.py
|   |-- learn_reader.py
|   `-- content_processor.py
|-- services/
|   |-- cache.py
|   `-- knowledge_generator.py
|-- tests/
|-- requirements.txt
`-- README.md
```

## Setup

Python 3.11 or later is required.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

No API key or cloud AI permission is required.

## Run

```powershell
python -m streamlit run app.py
```

Open `http://localhost:8501`.

## Example

Enter:

```text
Azure Functions Consumption versus Premium hosting
```

The application generates several focused Learn searches, retrieves the best official pages, and displays extracted summary sentences, relevant sections, code examples when present, and the exact source links.

## How It Works

`tools/learn_search.py` calls the public Microsoft Learn search endpoint, deduplicates results, and rejects all external domains.

`tools/learn_reader.py` validates the URL before every request and validates the final URL after redirects. It removes site navigation and unrelated interface elements before extracting documentation text.

`agent/orchestrator.py` decides whether a topic needs one search or several focused searches, ranks results, retrieves pages, and verifies references.

`services/knowledge_generator.py` creates a structured answer by ranking source sentences and organizing relevant documentation sections.

## Tests

```powershell
python -m pytest -q
```

Tests cover URL security, search parsing, duplicate removal, timeouts, HTML extraction, chunk limits, extractive generation, structured output, and the complete extraction workflow. Tests do not require network access.