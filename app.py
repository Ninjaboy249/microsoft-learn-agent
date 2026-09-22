"""Streamlit UI for the Microsoft Learn Knowledge Generator Agent."""

from __future__ import annotations

from html import escape
import re

import streamlit as st
from pydantic import ValidationError

from agent.models import LearnResponse
from agent.orchestrator import AgentError, MicrosoftLearnAgent
from services.knowledge_generator import ExtractiveKnowledgeGenerator

MODES = [
    "Quick Summary",
    "Study Notes",
    "Beginner Explanation",
    "Deep Dive",
    "Interview Preparation",
    "Step-by-Step Guide",
    "Quiz",
]

st.set_page_config(page_title="Microsoft Learn Knowledge Generator", page_icon="📘", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    :root {
        --ink: #15251f;
        --muted: #5d6d66;
        --brand: #006b5f;
        --brand-deep: #004d45;
        --accent: #d9a441;
        --surface: #ffffff;
        --surface-soft: #f3f8f6;
        --line: #cfdcd7;
    }
    .stApp {
        background: #ffffff;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(255, 255, 255, .94); backdrop-filter: blur(12px); }
    [data-testid="stMainBlockContainer"] { max-width: 1240px; padding-top: 2rem; padding-bottom: 4rem; }
    html, body, .stApp { font-family: 'DM Sans', sans-serif; }
    [data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
    h1, h2, h3 { color: var(--ink); font-family: 'Manrope', sans-serif; letter-spacing: 0; }
    .hero { border-bottom: 1px solid var(--line); padding: 1.25rem 0 1.5rem; margin-bottom: 1.5rem; }
    .hero-mark { color: var(--brand); font-size: .76rem; font-weight: 800; letter-spacing: .12em; margin-bottom: .55rem; text-transform: uppercase; }
    .hero h1 { font-size: clamp(2rem, 4vw, 3.4rem); line-height: 1.05; margin: 0; max-width: 820px; }
    .hero p { color: var(--muted); font-size: 1.05rem; margin: .65rem 0 0; max-width: 700px; }
    .result-heading { align-items: end; border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; margin: 2rem 0 1rem; padding-bottom: .8rem; }
    .result-heading strong { font: 700 1.25rem 'Manrope', sans-serif; }
    .result-heading span { color: var(--muted); font-size: .82rem; }
    .result-label { color: var(--brand); font: 800 .74rem 'DM Sans'; letter-spacing: .1em; margin: 1.6rem 0 .55rem; text-transform: uppercase; }
    .summary-band { background: #ffffff; border: 1px solid var(--line); border-left: 4px solid var(--brand); border-radius: 0 6px 6px 0; color: #18372f; font-size: 1.05rem; line-height: 1.75; padding: 1rem 1.15rem; }
    .source-link { background: var(--surface); border: 1px solid var(--line); border-left: 4px solid var(--brand); border-radius: 0 6px 6px 0; color: var(--ink) !important; display: block; margin: .65rem 0; padding: .85rem 1rem; text-decoration: none !important; }
    .source-link:hover { border-color: var(--brand); box-shadow: 0 5px 18px rgba(16, 62, 51, .08); }
    .source-link small { color: var(--muted); display: block; margin-top: .25rem; overflow-wrap: anywhere; }
    .doc-breadcrumb { color: var(--muted); font-size: .92rem; margin: .4rem 0 1.8rem; }
    .doc-breadcrumb span { color: var(--brand); }
    .doc-nav-title { color: var(--ink); font: 700 1rem 'Manrope', sans-serif; margin-bottom: .8rem; }
    .doc-nav-label { color: var(--muted); font-size: .72rem; font-weight: 800; letter-spacing: .1em; margin: 1rem 0 .45rem; text-transform: uppercase; }
    .doc-nav-link { border-left: 2px solid transparent; color: var(--muted) !important; display: block; font-size: .9rem; padding: .42rem .7rem; text-decoration: none !important; }
    .doc-nav-link:hover { background: #ffffff; border-left-color: var(--brand); color: var(--brand) !important; }
    .doc-title {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.85rem, 2.5vw, 2.4rem) !important;
        font-weight: 750;
        line-height: 1.2 !important;
        margin: 0 0 1.25rem;
        max-width: 28ch;
    }
    .premium-header {
        --header-accent: #006b5f;
        align-items: stretch;
        background: #f6faf8;
        border-bottom: 1px solid var(--line);
        border-top: 4px solid var(--header-accent);
        display: grid;
        margin: -1.5rem -1.5rem 1.25rem;
        min-height: 230px;
        overflow: hidden;
    }
    .premium-header.with-visual { grid-template-columns: minmax(0, 1fr) minmax(190px, 34%); }
    .premium-header.family-azure { --header-accent: #0078d4; background: #f4f9fd; }
    .premium-header.family-developer { --header-accent: #512bd4; background: #f8f7fd; }
    .premium-header.family-productivity { --header-accent: #d83b01; background: #fff8f5; }
    .premium-header.family-security { --header-accent: #107c10; background: #f5faf5; }
    .premium-header.family-windows { --header-accent: #0067b8; background: #f4f9fc; }
    .premium-copy { align-self: center; padding: clamp(1.5rem, 3vw, 2.4rem); }
    .premium-kicker { align-items: center; color: var(--header-accent); display: flex; font-size: .76rem; font-weight: 800; gap: .45rem; letter-spacing: .08em; margin-bottom: .85rem; text-transform: uppercase; }
    .premium-kicker [data-testid="stIconMaterial"] { font-size: 1.15rem; letter-spacing: 0; }
    .premium-meta { color: var(--muted); font-size: .9rem; font-weight: 600; margin-top: .9rem; }
    .premium-visual { align-items: center; background: #ffffff; border-left: 1px solid var(--line); display: flex; justify-content: center; min-height: 226px; overflow: hidden; }
    .premium-visual img { height: 100%; max-height: 300px; object-fit: contain; padding: 1rem; width: 100%; }
    .premium-visual.fallback { background: var(--header-accent); color: #ffffff; }
    .premium-visual.fallback [data-testid="stIconMaterial"] { font-size: 4.5rem; font-variation-settings: 'FILL' 0, 'wght' 300; }
    .doc-summary { color: var(--ink); font-size: 1.08rem; line-height: 1.8; margin: 1.25rem 0; max-width: 72ch; }
    .doc-summary p { margin: 0 0 1rem; }
    .doc-summary p:last-child { margin-bottom: 0; }
    .section-anchor { border-top: 1px solid var(--line); margin-top: 2.4rem; padding-top: .25rem; }
    .definition-callout { background: #ffffff; border: 1px solid #5ca6d6; border-left: 4px solid #0078d4; border-radius: 0 6px 6px 0; color: var(--ink); margin: 1.4rem 0; padding: 1rem 1.15rem; }
    .definition-callout strong { display: block; font: 700 1.15rem 'Manrope', sans-serif; margin-bottom: .55rem; }
    .definition-callout p { color: var(--ink) !important; font-size: 1.05rem; line-height: 1.7; margin: 0; }
    .doc-rule { border-top: 1px solid var(--line); margin: 1.4rem 0; }
    .doc-source { color: #0067b8 !important; overflow-wrap: anywhere; text-decoration: none !important; }
    [data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"], [data-testid="stSelectbox"] div[role="group"] {
        background: var(--surface) !important;
        border-color: #9fb8af !important;
        color: var(--ink) !important;
    }
    [data-testid="stTextInputField"], [data-testid="stNumberInputField"], [data-testid="stSelectbox"] input {
        color: var(--ink) !important;
        caret-color: var(--brand) !important;
    }
    [data-testid="stTextInputField"]::placeholder { color: #71827b !important; opacity: 1; }
    [data-testid="stWidgetLabel"] p { color: #344a42 !important; font-weight: 700; }
    div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button,
    div[data-testid="stPopover"] button,
    div[data-testid="stLinkButton"] a {
        border-radius: 5px !important;
        font-weight: 700 !important;
        min-height: 2.75rem;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background: var(--brand) !important;
        border-color: var(--brand) !important;
        color: #ffffff !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover { background: var(--brand-deep) !important; }
    div[data-testid="stButton"] button[kind="secondary"], div[data-testid="stDownloadButton"] button,
    div[data-testid="stPopover"] button,
    div[data-testid="stLinkButton"] a {
        background: var(--surface) !important;
        border-color: #9fb8af !important;
        color: #163a32 !important;
    }
    div[data-testid="stButton"] button[kind="secondary"] p, div[data-testid="stButton"] button[kind="secondary"] span,
    div[data-testid="stDownloadButton"] button p, div[data-testid="stDownloadButton"] button span,
    div[data-testid="stPopover"] button p, div[data-testid="stPopover"] button span,
    div[data-testid="stLinkButton"] a p, div[data-testid="stLinkButton"] a span { color: inherit !important; }
    div[data-testid="stButton"] button[kind="secondary"]:hover, div[data-testid="stDownloadButton"] button:hover,
    div[data-testid="stPopover"] button:hover,
    div[data-testid="stLinkButton"] a:hover {
        background: #e7f2ee !important;
        border-color: var(--brand) !important;
    }
    div[data-testid="stButton"] button:disabled {
        background: #e8eeeb !important;
        border-color: #c9d4d0 !important;
        color: #71827b !important;
        opacity: 1 !important;
    }
    div[data-testid="stButton"] button:disabled p, div[data-testid="stButton"] button:disabled span { color: inherit !important; }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] { border-bottom: 1px solid var(--line); gap: 1.2rem; }
    div[data-testid="stTabs"] button[role="tab"] { color: var(--muted); font-weight: 700; padding-left: .2rem; padding-right: .2rem; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--brand); }
    div[data-testid="stCode"] { background: #f7f9f8; border: 1px solid var(--line); border-radius: 6px; }
    div[data-testid="stCode"] > div:last-child,
    div[data-testid="stCode"] [data-testid="stElementToolbarButton"] {
        opacity: 1 !important;
        visibility: visible !important;
    }
    div[data-testid="stCode"] [aria-label="Copy to clipboard"] {
        background: #ffffff !important;
        border: 1px solid #9fb8af !important;
        color: var(--ink) !important;
    }
    [data-testid="stSidebar"] { background: #ffffff; border-left: 1px solid var(--line); }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
        .hero { padding-top: .5rem; }
        .result-heading { align-items: start; flex-direction: column; gap: .25rem; }
    }
    </style>
    <div class="hero">
      <div class="hero-mark">Official docs, focused answers</div>
      <h1>Microsoft Learn Knowledge Generator</h1>
      <p>Turn Microsoft Learn documentation into focused summaries, study notes, and practical references.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    defaults = {"topic": "", "mode": "Study Notes", "source_count": 5, "response": None, "history": []}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def get_agent() -> MicrosoftLearnAgent:
    return MicrosoftLearnAgent(knowledge_generator=ExtractiveKnowledgeGenerator())


def clear_workspace() -> None:
    st.session_state.topic = ""
    st.session_state.response = None


def load_history_topic() -> None:
    selected = st.session_state.get("history_topic")
    if selected:
        st.session_state.topic = selected
        st.session_state.response = None


def run_generation() -> None:
    topic = st.session_state.topic.strip()
    if not topic:
        st.warning("Enter a Microsoft technology topic to begin.")
        return
    try:
        with st.spinner("Searching, reading, and organizing Microsoft Learn documentation..."):
            result = get_agent().run(topic, st.session_state.mode, st.session_state.source_count)
        st.session_state.response = result.model_dump(mode="json")
        topics = [item for item in st.session_state.history if item.casefold() != topic.casefold()]
        st.session_state.history = [topic, *topics][:8]
    except AgentError as exc:
        st.error(str(exc))
    except Exception:
        st.error("The request could not be completed. Check your configuration and try again.")


def response_text(response: LearnResponse) -> str:
    definition = response.definition or first_sentence(response.summary)
    sections = [response.topic, "", "Definition", definition, "", "Overview", response.summary]
    if response.key_concepts:
        sections.extend(["", "Key Concepts", *[f"- {concept}" for concept in response.key_concepts]])
    for note in response.notes:
        sections.extend(["", note.heading, note.content])
    if response.sources:
        sections.extend(["", "Microsoft Learn Sources"])
        sections.extend(f"- {source.title}: {source.url}" for source in response.sources)
    return "\n".join(sections)


def section_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "section"


def first_sentence(value: str) -> str:
    match = re.search(r"^.*?[.!?](?:\s|$)", value.strip())
    return match.group(0).strip() if match else value.strip()


def code_language(value: str) -> str | None:
    if re.search(r"(^|\n)\s*(?:az|curl|sudo|brew|npm|pip|func|docker|kubectl)\b", value):
        return "bash"
    if re.search(r"(^|\n)\s*(?:Get-|Set-|New-|Remove-|Install-|\$[A-Za-z_])", value):
        return "powershell"
    if value.lstrip().startswith(("{", "[")):
        return "json"
    return None


def readable_markdown(value: str, sentences_per_paragraph: int = 2) -> str:
    """Add visual breathing room without changing extracted wording."""
    blocks: list[str] = []
    prose: list[str] = []

    def flush_prose() -> None:
        if not prose:
            return
        text = " ".join(prose)
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        blocks.extend(
            " ".join(sentences[index : index + sentences_per_paragraph])
            for index in range(0, len(sentences), sentences_per_paragraph)
        )
        prose.clear()

    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            flush_prose()
        elif re.match(r"^(?:[-*+] |\d+[.)] )", line):
            flush_prose()
            if blocks and re.match(r"^(?:[-*+] |\d+[.)] )", blocks[-1]):
                blocks[-1] = f"{blocks[-1]}\n{line}"
            else:
                blocks.append(line)
        else:
            prose.append(line)
    flush_prose()
    return "\n\n".join(blocks)


def result_family(response: LearnResponse) -> tuple[str, str, str]:
    source_url = str(response.sources[0].url) if response.sources else ""
    text = f"{response.topic} {source_url}".casefold()
    families = (
        (("power bi", "power platform", "microsoft 365", "teams", "sharepoint"), "Microsoft 365", "workspaces", "productivity"),
        (("security", "entra", "defender", "sentinel", "identity"), "Microsoft Security", "shield", "security"),
        ((".net", "dotnet", "c#", "asp.net", "visual studio", "github"), "Developer platform", "code_blocks", "developer"),
        (("windows", "winui"), "Windows", "desktop_windows", "windows"),
        (("azure",), "Microsoft Azure", "cloud", "azure"),
    )
    for terms, label, icon, family in families:
        if any(term in text for term in terms):
            return label, icon, family
    return "Microsoft Learn", "auto_stories", "learn"


def render_response(response: LearnResponse) -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #ffffff !important; background-image: none !important; color: var(--ink) !important; }
        [data-testid="stHeader"] { background: rgba(255, 255, 255, .94) !important; }
        [data-testid="stMainBlockContainer"] { max-width: 1480px; padding-top: 4.75rem; }
        .hero { display: none; }
        .hero-mark { color: var(--brand); }
        .hero h1, .hero p, h1, h2, h3, [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li { color: var(--ink); }
        [data-testid="stWidgetLabel"] p { color: #344a42 !important; }
        [data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"],
        [data-testid="stSelectbox"] div[role="group"] { background: #ffffff !important; border-color: #9fb8af !important; }
        [data-testid="stTextInputField"], [data-testid="stNumberInputField"], [data-testid="stSelectbox"] input { color: var(--ink) !important; }
        [data-testid="stSidebar"] { background: #ffffff !important; border-color: var(--line) !important; }
        [data-testid="stHorizontalBlock"]:has(.doc-nav-title) > [data-testid="stColumn"]:nth-child(2) {
            background: #ffffff;
            border: 1px solid #d9e3df;
            border-radius: 8px;
            box-shadow: 0 18px 45px rgba(21, 37, 31, .08);
            overflow: hidden;
            padding: 1.5rem;
        }
        [data-testid="stHorizontalBlock"]:has(.doc-nav-title) > [data-testid="stColumn"]:nth-child(2) [data-testid="stMarkdownContainer"] { max-width: 72ch; }
        [data-testid="stHorizontalBlock"]:has(.doc-nav-title) > [data-testid="stColumn"]:nth-child(2) [data-testid="stMarkdownContainer"] p,
        [data-testid="stHorizontalBlock"]:has(.doc-nav-title) > [data-testid="stColumn"]:nth-child(2) [data-testid="stMarkdownContainer"] li {
            font-size: 1.02rem;
            line-height: 1.75;
        }
        [data-testid="stHorizontalBlock"]:has(.doc-nav-title) > [data-testid="stColumn"]:nth-child(2) [data-testid="stMarkdownContainer"] p { margin-bottom: 1rem; }
        [data-testid="stHorizontalBlock"]:has(.doc-nav-title) > [data-testid="stColumn"]:nth-child(2) [data-testid="stMarkdownContainer"] ul,
        [data-testid="stHorizontalBlock"]:has(.doc-nav-title) > [data-testid="stColumn"]:nth-child(2) [data-testid="stMarkdownContainer"] ol {
            margin: .45rem 0 1.15rem;
            padding-left: 1.4rem;
        }
        [data-testid="stHorizontalBlock"]:has(.doc-nav-title) > [data-testid="stColumn"]:nth-child(2) [data-testid="stMarkdownContainer"] li {
            margin-bottom: .45rem;
            padding-left: .2rem;
        }
        div[data-testid="stButton"] button[kind="secondary"], div[data-testid="stDownloadButton"] button,
        div[data-testid="stPopover"] button,
        div[data-testid="stLinkButton"] a {
            background: #ffffff !important; border-color: #9fb8af !important; color: #163a32 !important;
        }
        div[data-testid="stButton"] button[kind="secondary"]:hover, div[data-testid="stDownloadButton"] button:hover,
        div[data-testid="stPopover"] button:hover,
        div[data-testid="stLinkButton"] a:hover {
            background: #ffffff !important; border-color: var(--brand) !important;
        }
        [data-testid="stColumn"]:has(.doc-nav-title) { align-self: flex-start; position: sticky; top: 4rem; }
        @media (max-width: 900px) {
            [data-testid="stColumn"]:has(.doc-nav-title) { display: none; }
            [data-testid="stHorizontalBlock"]:has(.doc-nav-title) [data-testid="stColumn"]:not(:has(.doc-nav-title)) {
                flex: 1 1 100% !important;
                width: 100% !important;
            }
            .doc-title { font-size: 2rem !important; margin-top: .5rem; }
            .premium-header { margin: -1.5rem -1.5rem 1.25rem; min-height: 0; }
            .premium-header.with-visual { grid-template-columns: 1fr; }
            .premium-visual { border-left: 0; border-top: 1px solid var(--line); max-height: 230px; min-height: 180px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    article_title = response.sources[0].title if response.sources else response.topic
    article_notes = [note for note in response.notes if note.heading.casefold() not in {"definition", "overview"}]
    note_links = "".join(
        f'<a class="doc-nav-link" href="#{section_id(note.heading)}">{escape(note.heading)}</a>'
        for note in article_notes
    )
    definition = response.definition or first_sentence(response.summary)
    overview = response.summary[len(definition):].strip() if response.summary.startswith(definition) else response.summary
    text = response_text(response)
    family_label, family_icon, family_class = result_family(response)
    image_source = next((source for source in response.sources if source.image_url), None)
    visual = (
        f'<div class="premium-visual"><img src="{escape(str(image_source.image_url), quote=True)}" '
        f'alt="{escape(image_source.title, quote=True)}"></div>'
        if image_source
        else f'<div class="premium-visual fallback"><span data-testid="stIconMaterial">{family_icon}</span></div>'
    )
    header_class = f"premium-header family-{family_class} with-visual"
    source_label = f"{len(response.sources)} official source{'s' if len(response.sources) != 1 else ''}"

    st.markdown(
        f'<div class="doc-breadcrumb"><span>Learn</span> &nbsp;›&nbsp; Microsoft Azure &nbsp;›&nbsp; {escape(response.topic)}</div>',
        unsafe_allow_html=True,
    )
    topic_nav, article, article_nav = st.columns([1.15, 3.5, 1.2], gap="large")

    with topic_nav:
        st.markdown(
            f'<div class="doc-nav-title">{escape(response.topic)}</div>'
            f'<div class="doc-nav-label">Overview</div>'
            f'<a class="doc-nav-link" href="#definition">Definition</a>'
            f'<a class="doc-nav-link" href="#article-summary">Overview</a>{note_links}'
            f'<a class="doc-nav-link" href="#references">References</a>',
            unsafe_allow_html=True,
        )

    with article:
        st.markdown(
            f'<div class="{header_class}"><div class="premium-copy">'
            f'<div class="premium-kicker"><span data-testid="stIconMaterial">verified</span>{family_label} learning brief</div>'
            f'<h1 class="doc-title">{escape(article_title)}</h1>'
            f'<div class="premium-meta">{source_label} · '
            f'{len(article_notes) + 3} focused sections</div></div>{visual}</div>',
            unsafe_allow_html=True,
        )
        action_source, action_copy, action_download = st.columns([1.5, 1, 1.2])
        with action_source:
            if response.sources:
                st.link_button(
                    "Open on Microsoft Learn",
                    str(response.sources[0].url),
                    icon=":material/open_in_new:",
                    use_container_width=True,
                )
        with action_copy:
            with st.popover("Copy", icon=":material/content_copy:", use_container_width=True):
                st.code(text, language=None, wrap_lines=True)
        with action_download:
            st.download_button(
                "Download",
                data=text,
                file_name="microsoft-learn-notes.txt",
                mime="text/plain",
                icon=":material/download:",
                use_container_width=True,
                on_click="ignore",
            )

        st.markdown('<div id="definition" class="doc-rule"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="definition-callout"><strong>Definition</strong><p>{escape(definition)}</p></div>',
            unsafe_allow_html=True,
        )
        if overview:
            st.markdown('<div id="article-summary"></div>', unsafe_allow_html=True)
            st.header("Overview")
            overview_html = "".join(
                f"<p>{escape(paragraph)}</p>" for paragraph in readable_markdown(overview).split("\n\n")
            )
            st.markdown(f'<div class="doc-summary">{overview_html}</div>', unsafe_allow_html=True)

        if response.key_concepts:
            st.markdown('<div id="key-capabilities" class="section-anchor"></div>', unsafe_allow_html=True)
            st.header("Key capabilities")
            st.markdown("\n".join(f"- {concept}" for concept in response.key_concepts))

        for note in article_notes:
            st.markdown(
                f'<div id="{section_id(note.heading)}" class="section-anchor"></div>',
                unsafe_allow_html=True,
            )
            st.header(note.heading)
            st.markdown(readable_markdown(note.content))

        if response.examples:
            st.markdown('<div id="examples" class="section-anchor"></div>', unsafe_allow_html=True)
            st.header("Examples")
            for example in response.examples:
                st.code(example, language=code_language(example), wrap_lines=False)

        st.markdown('<div id="references" class="section-anchor"></div>', unsafe_allow_html=True)
        st.header("References")
        if response.sources:
            for source in response.sources:
                safe_url = escape(str(source.url), quote=True)
                st.markdown(
                    f'<a class="doc-source" href="{safe_url}" target="_blank">{escape(source.title)} ↗</a>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No retrieved references are available for this response.")

    with article_nav:
        st.markdown(
            '<div class="doc-nav-title">In this answer</div>'
            '<a class="doc-nav-link" href="#definition">Definition</a>'
            + ('<a class="doc-nav-link" href="#article-summary">Overview</a>' if overview else '')
            + ('<a class="doc-nav-link" href="#key-capabilities">Key capabilities</a>' if response.key_concepts else '')
            + note_links
            + ('<a class="doc-nav-link" href="#examples">Examples</a>' if response.examples else '')
            + '<a class="doc-nav-link" href="#references">References</a>',
            unsafe_allow_html=True,
        )


initialize_state()

if st.session_state.response:
    query_col, mode_col, source_col, generate_col, regenerate_col, clear_col = st.columns([3, 1.5, .65, 1.15, 1.15, .8])
    with query_col:
        st.text_input("Topic", key="topic", placeholder="Search Microsoft Learn", label_visibility="collapsed")
    with mode_col:
        st.selectbox("Generation mode", MODES, key="mode", label_visibility="collapsed")
    with source_col:
        st.number_input("Sources", min_value=1, max_value=8, key="source_count", label_visibility="collapsed")
    with generate_col:
        generate = st.button("Search", type="primary", icon=":material/search:", use_container_width=True)
    with regenerate_col:
        regenerate = st.button("Regenerate", icon=":material/refresh:", use_container_width=True)
    with clear_col:
        st.button("Clear", icon=":material/delete_sweep:", on_click=clear_workspace, use_container_width=True)

    if generate or regenerate:
        run_generation()
        if st.session_state.response:
            st.rerun()

    try:
        render_response(LearnResponse.model_validate(st.session_state.response))
    except ValidationError:
        st.error("The saved response is invalid. Generate it again.")
else:
    main, sidebar = st.columns([3, 1], gap="large")
    with main:
        st.text_input("What do you want to learn?", key="topic", placeholder="Azure Functions")
        mode_col, source_col = st.columns([2, 1])
        with mode_col:
            st.selectbox("Generation mode", MODES, key="mode")
        with source_col:
            st.number_input("Number of sources", min_value=1, max_value=8, key="source_count")

        generate_col, regenerate_col, clear_col = st.columns([2, 1.25, 1])
        with generate_col:
            generate = st.button("Generate answer", type="primary", icon=":material/search:", use_container_width=True)
        with regenerate_col:
            regenerate = st.button("Regenerate", icon=":material/refresh:", disabled=True, use_container_width=True)
        with clear_col:
            st.button("Clear", icon=":material/delete_sweep:", on_click=clear_workspace, use_container_width=True)

        if generate:
            run_generation()
            if st.session_state.response:
                st.rerun()

    with sidebar:
        st.subheader("Workspace")
        st.caption("Recent topics")
        if st.session_state.history:
            st.selectbox(
                "Open a previous topic",
                ["", *st.session_state.history],
                key="history_topic",
                on_change=load_history_topic,
                label_visibility="collapsed",
            )
        else:
            st.caption("Generated topics appear here during this session.")
        st.divider()
        st.caption("Knowledge source")
        st.markdown("**learn.microsoft.com only**")
        st.caption("Answer generation")
        st.markdown("**Built-in extractive mode**")