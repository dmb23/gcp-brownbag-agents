import asyncio
import functools
import re
from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, Optional, TypeVar

import anyio
import anyio.to_thread
import httpx
from duckduckgo_search import DDGS
from loguru import logger
from markdownify import markdownify
from pydantic import TypeAdapter, ValidationError
from pydantic_ai import RunContext
from pydantic_ai.tools import Tool

from gcp_brownbag_agents import types

# Type variable for the dependencies
T = TypeVar("T")


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
        # Get class name and docstring for tool name and description
        tool_name = self.__class__.__name__
        tool_description = self.__class__.__doc__ or f"A {tool_name} instance"

        return Tool(
            self.execute,
            takes_ctx=self.takes_ctx(),
            prepare=self.prepare_func,
            name=tool_name,
            description=tool_description,
        )

    @abstractmethod
    def takes_ctx(self) -> bool:
        """Return whether this tool takes a context argument."""
        pass


class HackerNewsTool(BaseTool[types.RunDeps]):
    """Tool for retrieving stories from HackerNews.

    API docs at https://github.com/HackerNews/API
    """

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
            ctx: The run context, providing an AsyncClient
            num_entries: maximum number of top stories to retrieve
            story_type: select if stories should be taken from a list of new stories / best stories / top (trending) stories
        """
        num_entries = min(num_entries, 500)
        url = f"https://hacker-news.firebaseio.com/v0/{story_type}stories.json"

        try:
            response = await ctx.deps.client.get(url)
            response.raise_for_status()
        except httpx.RequestError as exc:
            logger.error(f"An error occurred while requesting {exc.request.url!r}.")
            return []
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"Error response {exc.response.status_code} while requesting {exc.request.url!r}."
            )
            return []

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

    async def _get_story(
        self, client: httpx.AsyncClient, id: str
    ) -> types.Story | None:
        try:
            r = await client.get(
                f"https://hacker-news.firebaseio.com/v0/item/{id}.json"
            )
            r.raise_for_status()
        except httpx.RequestError as exc:
            logger.error(f"An error occurred while requesting {exc.request.url!r}.")
            return None
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"Error response {exc.response.status_code} while requesting {exc.request.url!r}."
            )
            return None

        try:
            return types.Story.model_validate_json(r.content)
        except ValidationError:
            return None

    def takes_ctx(self) -> bool:
        return True


class DuckDuckGoSearchTool(BaseTool):
    """Tool for searching with DuckDuckGo."""

    def __init__(
        self,
        client: Optional[DDGS] = None,
        max_results: Optional[int] = None,
        prepare_func=None,
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


class WebpageTool(BaseTool[types.RunDeps]):
    """Tool for visiting webpages and retrieving their content.

    This tool will only work on HTML pages, and is incapable of parsing PDFs."""

    def __init__(self, prepare_func=None):
        """Initialize the webpage tool."""
        super().__init__(prepare_func)

    async def execute(self, ctx: RunContext[types.RunDeps], url: str) -> str:
        """Visits a webpage at the given URL and returns its content as a markdown string.

        Args:
            ctx: The run context, providing an AsyncClient
            url: The URL of the webpage to visit.

        Returns:
            The content of the webpage converted to Markdown, or an error message if the request fails.
        """
        if url.lower().endswith(".pdf"):
            return "WebpageTool cannot be used to read PDF files!"

        try:
            # Send a GET request to the URL
            response = await ctx.deps.client.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Convert the HTML content to Markdown
            markdown_content = markdownify(response.text).strip()

            # Remove multiple line breaks
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

            return markdown_content

        except httpx.RequestError as exc:
            return f"An error occurred while requesting {exc.request.url!r}."
        except httpx.HTTPStatusError as exc:
            return f"Error response {exc.response.status_code} while requesting {exc.request.url!r}."
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

    def takes_ctx(self) -> bool:
        return True
