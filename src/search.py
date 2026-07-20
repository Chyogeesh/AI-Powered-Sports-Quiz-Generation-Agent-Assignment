"""
Live web search using DuckDuckGo (no API key required).

Note: the `duckduckgo-search` package was renamed to `ddgs` in newer releases.
This module tries both import paths so it keeps working regardless of which
version pip installs.
"""

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


def get_live_news_context(sport_name: str, max_results: int = 3) -> str:
    """
    Searches the live web for recent news/results for the given sport.
    Returns a single formatted text block of the top snippets, or a
    graceful fallback message if the search fails (e.g. rate limiting,
    no network access).
    """
    search_query = f"{sport_name} latest tournament results championship winners news 2026"
    retrieved_texts = []

    try:
        with DDGS() as ddgs:
            results = ddgs.text(search_query, max_results=max_results)
            for index, r in enumerate(results, start=1):
                title = r.get("title", "No Title")
                snippet = r.get("body", "No snippet available")
                retrieved_texts.append(f"Web Source {index}: {title}\nSnippet: {snippet}")
    except Exception as e:
        print(f"[search] Web search failed or was rate-limited: {e}")
        return "No live web results available right now. Relying on offline historic facts only."

    if not retrieved_texts:
        return "No live web results found for this query."

    return "\n\n".join(retrieved_texts)
