import asyncio
import functools
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional, TypeVar, Generic, Any

import anyio
import anyio.to_thread
import httpx
from duckduckgo_search import DDGS
from markdownify import markdownify
from pydantic import TypeAdapter, ValidationError
from pydantic_ai import RunContext
from pydantic_ai.tools import Tool, ToolDefinition

from gcp_brownbag_agents import types

# Type variable for the dependencies
T = TypeVar('T')

# https://github.com/HackerNews/API

class BaseTool(ABC, Generic[T]):
    """Base class for all tools."""
    
    def __init__(self, prepare_func=None):
        """Initialize the tool with an optional prepare function."""
        self.prepare_func = prepare_func
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute the tool functionality."""
        pass
    
    def get_tool(self) -> Tool:
        """Return a Tool instance for this tool."""
        return Tool(
            self.execute,
            takes_ctx=self.takes_ctx(),
            prepare=self.prepare_func
        )
    
    @abstractmethod
    def takes_ctx(self) -> bool:
        """Return whether this tool takes a context argument."""
        pass


class HackerNewsTool(BaseTool[types.RunDeps]):
    """Tool for retrieving stories from HackerNews."""
    
    def __init__(self, prepare_func=None):
        """Initialize the HackerNews tool."""
        super().__init__(prepare_func)
    
    async def execute(
        self,
        ctx: RunContext[types.RunDeps],
        num_entries: int,
        story_type: Literal["top", "best", "new"] = "top",
    ) -> list[types.Story]:
        """Retrieve the top stories from HackerNews

        Args:
            ctx: The run context
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
                tg.create_task(self._get_story(client=ctx.deps.client, id=id))
                for id in stories_ids
            ]

        # do it in two steps so that pydantic understands that None is filtered
        stories = [t.result() for t in tasks]
        stories = [s for s in stories if s]

        return stories
    
    async def _get_story(self, client: httpx.AsyncClient, id: str) -> types.Story | None:
        r = await client.get(f"https://hacker-news.firebaseio.com/v0/item/{id}.json")
        try:
            return types.Story.model_validate_json(r.content)
        except ValidationError:
            return None
    
    def takes_ctx(self) -> bool:
        return True


async def select_hn(
    ctx: RunContext[types.RunDeps], tool_def: ToolDefinition
) -> ToolDefinition | None:
    if ctx.deps.search_goal == "HN":
        return tool_def
    return None


class DuckDuckGoSearchTool(BaseTool):
    """Tool for searching with DuckDuckGo."""
    
    def __init__(
        self, 
        client: Optional[DDGS] = None,
        max_results: Optional[int] = None,
        prepare_func=None
    ):
        """Initialize the DuckDuckGo search tool.
        
        Args:
            client: The DuckDuckGo search client
            max_results: The maximum number of results. If None, returns results only from the first response
            prepare_func: Function to determine if this tool should be used
        """
        super().__init__(prepare_func)
        self.client = client or DDGS()
        self.max_results = max_results
        self.duckduckgo_ta = TypeAdapter(list[types.DuckDuckGoResult])
    
    async def execute(self, query: str) -> list[types.DuckDuckGoResult]:
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
        return self.duckduckgo_ta.validate_python(results)
    
    def takes_ctx(self) -> bool:
        return False


async def select_search(
    ctx: RunContext[types.RunDeps], tool_def: ToolDefinition
) -> ToolDefinition | None:
    if ctx.deps.search_goal != "HN":
        return tool_def
    return None


class WebpageTool(BaseTool[types.RunDeps]):
    """Tool for visiting webpages and retrieving their content."""
    
    def __init__(self, prepare_func=None):
        """Initialize the webpage tool."""
        super().__init__(prepare_func)
    
    async def execute(self, ctx: RunContext[types.RunDeps], url: str) -> str:
        """Visits a webpage at the given URL and returns its content as a markdown string.

        Args:
            ctx: The run context
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
    
    def takes_ctx(self) -> bool:
        return True


# Tool classes are now used directly
