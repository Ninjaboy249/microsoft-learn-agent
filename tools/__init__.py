"""Custom Microsoft Learn retrieval tools."""

from .learn_reader import get_microsoft_learn_page
from .learn_search import search_microsoft_learn

__all__ = ["get_microsoft_learn_page", "search_microsoft_learn"]