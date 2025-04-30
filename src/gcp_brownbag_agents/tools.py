import asyncio
import functools
import re
from dataclasses import dataclass, field
from typing import Literal

import anyio
import anyio.to_thread
import httpx
from duckduckgo_search import DDGS
from markdownify import markdownify
from pydantic import TypeAdapter, ValidationError
from pydantic_ai import RunContext
from pydantic_ai.tools import Tool, ToolDefinition

from gcp_brownbag_agents import types

# https://github.com/HackerNews/API


async def _get_story(client: httpx.AsyncClient, id: str) -> types.Story | None:
    r = await client.get(f"https://hacker-news.firebaseio.com/v0/item/{id}.json")
    try:
        return types.Story.model_validate_json(r.content)
    except ValidationError:
        return None


async def _hacker_news_tool_function(
    ctx: RunContext[types.RunDeps],
    num_entries: int,
    story_type: Literal["top", "best", "new"] = "top",
) -> list[types.Story]:
    """Retrieve the top stories from HackerNews

    Args:
        num_entries: maximum number of top stories to retrieve
        story_type: select if stories should be taken from a list of new stories / best stories / top (trending) stories
    """
    num_entries = min(num_entries, 500)
    url = f"https://hacker-news.firebaseio.com/v0/{story_type}stories.json"

    response = await ctx.deps.client.get(url)
    response.raise_for_status()

    stories_ids = response.json()[:num_entries]

    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(_get_story(client=ctx.deps.client, id=id))
            for id in stories_ids
        ]

    # do it in two steps so that pydantic understands that None is filtered
    stories = [t.result() for t in tasks]
    stories = [s for s in stories if s]

    return stories


async def select_hn(
    ctx: RunContext[types.RunDeps], tool_def: ToolDefinition
) -> ToolDefinition | None:
    if ctx.deps.search_goal == "HN":
        return tool_def
    return None


def hacker_news_tool() -> Tool[types.RunDeps]:
    return Tool(_hacker_news_tool_function, takes_ctx=True, prepare=select_hn)


# copied from pydantic_ai to adjust for Python<3.12
duckduckgo_ta = TypeAdapter(list[types.DuckDuckGoResult])


@dataclass
class DuckDuckGoSearchTool:
    """The DuckDuckGo search tool."""

    client: DDGS = field(default_factory=DDGS)
    """The DuckDuckGo search client."""

    max_results: int | None = None
    """The maximum number of results. If None, returns results only from the first response."""

    async def __call__(self, query: str) -> list[types.DuckDuckGoResult]:
        """Searches DuckDuckGo for the given query and returns the results.

        Args:
            query: The query to search for.

        Returns:
            The search results.
        """
        search = functools.partial(self.client.text, max_results=self.max_results)
        results = await anyio.to_thread.run_sync(search, query)
        if len(results) == 0:
            raise RuntimeError("No search results found.")
        return duckduckgo_ta.validate_python(results)


async def select_search(
    ctx: RunContext[types.RunDeps], tool_def: ToolDefinition
) -> ToolDefinition | None:
    if ctx.deps.search_goal != "HN":
        return tool_def
    return None


def duckduckgo_search_tool(*args, **kwargs) -> Tool:
    _prepare = kwargs.pop("prepare", select_search)
    return Tool(
        DuckDuckGoSearchTool(*args, **kwargs).__call__,
        takes_ctx=False,
        prepare=_prepare,
    )


# basic idea from smolagents
async def _visit_webpage_tool_function(ctx: RunContext[types.RunDeps], url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = await ctx.deps.client.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except httpx.HTTPError as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


def visit_webpage_tool() -> Tool[types.RunDeps]:
    return Tool(_visit_webpage_tool_function, takes_ctx=True)
