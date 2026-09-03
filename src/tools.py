"""
Tool functions for the agentic RAG loop.

Each function takes structured arguments and returns a plain string
that gets fed back to the LLM as a tool result.

Tools:
  - local_search(query)              — hybrid search over local vector index
  - wikipedia_search(query)          — live Wikipedia MediaWiki API
  - lyrics_search(artist, track)     — LRCLIB free lyrics API
"""

import re
from dataclasses import dataclass
from pathlib import Path

import requests

# ── local search imports (resolved relative to src/) ─────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from search import hybrid_search

# ── constants ─────────────────────────────────────────────────────────────────
WIKI_API = "https://en.wikipedia.org/w/api.php"
LRCLIB_API = "https://lrclib.net/api/get"
HEADERS = {
    "User-Agent": "MusicEncyclopediaRAG/1.0 (LLM Zoomcamp project; contact: eyademam28@gmail.com)"
}
MAX_WIKI_WORDS = 3000   # keep context window manageable


class ExternalToolServiceError(Exception):
    """Raised when a live tool provider cannot be reached safely."""


@dataclass(frozen=True)
class ToolResult:
    """Compact metadata for a tool result sent to the agent."""

    content: str
    source: str | None = None
    chunk_count: int = 0


# ── Tool: local_search ────────────────────────────────────────────────────────

def local_search(query: str, text_index, vector_index, embedder, num_results: int = 5) -> ToolResult:
    """Hybrid search over the local prebuilt knowledge base."""
    chunks = hybrid_search(text_index, vector_index, embedder, query, num_results=num_results)
    if not chunks:
        return ToolResult(f"No results found in local knowledge base for: '{query}'")

    parts = [f"[{c['topic']}] {c['chunk']}" for c in chunks]
    topics = list(dict.fromkeys(c["topic"] for c in chunks if c.get("topic")))
    source = ", ".join(topics) if topics else "Local Knowledge Base"
    return ToolResult("\n\n".join(parts), source=source, chunk_count=len(chunks))


# ── Tool: wikipedia_search ────────────────────────────────────────────────────

def _clean_wiki_text(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"={2,}.*?={2,}", "", text)  # strip == Section Headers ==
    return text.strip()


def wikipedia_search(query: str) -> ToolResult:
    """Fetch a Wikipedia article extract in real time."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": query,
        "format": "json",
        "redirects": 1,
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for _, page in pages.items():
            if "missing" in page:
                return ToolResult(f"No Wikipedia article found for '{query}'.")
            title = page.get("title", query)
            text = _clean_wiki_text(page.get("extract", ""))
            if not text:
                return ToolResult(f"Wikipedia article '{title}' contains no usable text.")
            # Truncate to MAX_WIKI_WORDS
            words = text.split()
            if len(words) > MAX_WIKI_WORDS:
                text = " ".join(words[:MAX_WIKI_WORDS]) + "\n\n[... article truncated ...]"
            return ToolResult(f"Wikipedia: {title}\n\n{text}", source=f"Wikipedia: {title}")
        return ToolResult(f"No Wikipedia article found for '{query}'.")
    except requests.RequestException as exc:
        raise ExternalToolServiceError(
            "Wikipedia is temporarily unavailable. Try another available tool if possible."
        ) from exc


# ── Tool: lyrics_search ───────────────────────────────────────────────────────

def lyrics_search(artist_name: str, track_name: str) -> ToolResult:
    """Fetch plain lyrics from LRCLIB (free, no API key required)."""
    params = {
        "artist_name": artist_name,
        "track_name": track_name,
    }
    try:
        r = requests.get(LRCLIB_API, params=params, headers=HEADERS, timeout=15)
        if r.status_code == 404:
            return ToolResult(f"No lyrics found for '{track_name}' by {artist_name}.")
        r.raise_for_status()
        data = r.json()
        plain = data.get("plainLyrics") or ""
        if not plain.strip():
            return ToolResult(
                f"Lyrics for '{track_name}' by {artist_name} were found but appear to be empty."
            )
        return ToolResult(
            f"Lyrics for '{track_name}' by {artist_name}:\n\n{plain}",
            source=f"LRCLIB: {track_name} by {artist_name}",
        )
    except requests.RequestException as exc:
        raise ExternalToolServiceError(
            "Lyrics search is temporarily unavailable. Try another available tool if possible."
        ) from exc


# ── OpenAI tool schemas ───────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "local_search",
            "description": (
                "Search the local music knowledge base for information about well-known "
                "artists, bands, and genres. Good for: The Beatles, Pink Floyd, Jazz, "
                "Hip Hop, Reggae, Beyoncé, Kendrick Lamar, David Bowie, etc. "
                "Use this as your first option for any music question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'Pink Floyd history' or 'origins of jazz'"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": (
                "Search Wikipedia for any music topic in real time. Use this for "
                "lesser-known artists, specific albums, music theory concepts, record labels, "
                "or when local_search doesn't return enough information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The Wikipedia article title or search term, e.g. 'Tame Impala' or 'Blue Note Records'"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lyrics_search",
            "description": (
                "Fetch the lyrics of a specific song. Use when the user asks about "
                "song lyrics, what a song means, or wants to read or quote lyrics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_name": {
                        "type": "string",
                        "description": "The artist or band name, e.g. 'Queen' or 'Bob Marley'"
                    },
                    "track_name": {
                        "type": "string",
                        "description": "The song/track name, e.g. 'Bohemian Rhapsody' or 'No Woman No Cry'"
                    }
                },
                "required": ["artist_name", "track_name"]
            }
        }
    }
]


# ── Dispatch ──────────────────────────────────────────────────────────────────

def execute_tool(name: str, arguments: dict, context: dict) -> ToolResult:
    """
    Route a tool call to the right function.

    Args:
        name:      Tool name from the LLM's tool_call
        arguments: Parsed JSON arguments from the LLM
        context:   Dict with 'text_index', 'vector_index', 'embedder' for local_search

    Returns:
        ToolResult with agent content, an optional successful-source label, and
        the number of local chunks retrieved.
    """
    if name == "local_search":
        query = arguments.get("query", "")
        return local_search(
            query,
            context["text_index"],
            context["vector_index"],
            context["embedder"],
        )

    elif name == "wikipedia_search":
        query = arguments.get("query", "")
        return wikipedia_search(query)

    elif name == "lyrics_search":
        artist = arguments.get("artist_name", "")
        track = arguments.get("track_name", "")
        return lyrics_search(artist, track)

    else:
        return ToolResult(f"Unknown tool: {name}")
