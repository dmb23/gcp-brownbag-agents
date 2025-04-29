import asyncio
import functools
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import anyio
import anyio.to_thread
import httpx
import logfire
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from markdownify import markdownify
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import Tool, ToolDefinition
from tenacity import retry, stop_after_attempt, wait_exponential
from typing_extensions import TypedDict


@dataclass
class RunDeps:
    client: httpx.AsyncClient
    search_goal: str


task = """Your task is to curate top stories from the website "HackerNews", identify the most relevant article for consultants working in a boutique data consultancy, extract detailed information from the selected article, and generate a comprehensive markdown report.

Guidelines:
1. Start with a lower number of stories from HackerNews, and decide if one of them is of high interest to consultants working on different data topics. If none of the retrieved stories sounds promising, continue to search through successive entries in HackerNews.
    i. Prioritize topics on Data Engineering, new promising Tools (preferrably in Python), MLOps or AI developments
2. Get detailed information from the url connected to the chosen story.
3. Generate a comprehensive Markdown report detailing your findings.
    i. structure your report into multiple sections. Focus on 
        a. a general overview of what the report is about
        b. the USP of the described tool / solution
        c. ways the described tool / solution can solve issues for possible customers of the consultancy
        c. ways the described tool / solution can ease the work of consultants
    ii. Keep the amount of text limited.
        a. prioritize single paragraphs with additional information in bullet points
        b. use mermaid diagrams where they help understanding
        c. link to figures from the website, if these help understanding
    iii. Provide Links to your sources
4. Return just a string containing the final markdown.

By following these instructions, you will create a valuable resource for consultants, keeping them informed about the latest trends and insights how to best help their customers with issues around data engineering, data science and AI. 
"""


class ReferenceLink(BaseModel):
    """Web address of a helpful resource"""

    description: str
    url: str


class ImageLink(BaseModel):
    """Web address of an image / figure explaining some context"""

    description: str
    url: str


class ResearchResult(BaseModel):
    """Results of some research towards a suitable topic for a presentation."""

    full_text: str
    references: list[ReferenceLink]
    images: list[ImageLink]


async def select_hn(
    ctx: RunContext[RunDeps], tool_def: ToolDefinition
) -> ToolDefinition | None:
    if ctx.deps.search_goal == "HN":
        return tool_def
    return None


async def select_search(
    ctx: RunContext[RunDeps], tool_def: ToolDefinition
) -> ToolDefinition | None:
    if ctx.deps.search_goal != "HN":
        return tool_def
    return None


class Story(BaseModel):
    title: str
    url: str


# https://github.com/HackerNews/API


async def _get_story(client: httpx.AsyncClient, id: str) -> Story | None:
    r = await client.get(f"https://hacker-news.firebaseio.com/v0/item/{id}.json")
    try:
        return Story.model_validate_json(r.content)
    except ValidationError:
        return None


async def hacker_news_tool(
    ctx: RunContext[RunDeps],
    num_entries: int,
    story_type: Literal["top", "best", "new"] = "top",
) -> list[Story]:
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


# copied from pydantic_ai to adjust for Python<3.12
class DuckDuckGoResult(TypedDict):
    """A DuckDuckGo search result."""

    title: str
    """The title of the search result."""
    href: str
    """The URL of the search result."""
    body: str
    """The body of the search result."""


duckduckgo_ta = TypeAdapter(list[DuckDuckGoResult])


@dataclass
class DuckDuckGoSearchTool:
    """The DuckDuckGo search tool."""

    client: DDGS
    """The DuckDuckGo search client."""

    max_results: int | None = None
    """The maximum number of results. If None, returns results only from the first response."""

    async def __call__(self, query: str) -> list[DuckDuckGoResult]:
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


# basic idea from smolagents
async def visit_webpage_tool(ctx: RunContext[RunDeps], url: str) -> str:
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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4))
async def get_research_result(
    research_agent: Agent[RunDeps, ResearchResult],
) -> ResearchResult:
    # can raise a status code 529 "Overloaded"!
    async with httpx.AsyncClient() as _client:
        research_deps = RunDeps(client=_client, search_goal="HN")
        run_result = await research_agent.run(task, deps=research_deps)

    return run_result.output


if __name__ == "__main__":
    load_dotenv()

    logfire.configure()
    logfire.instrument_httpx(capture_all=True)

    model_name = "anthropic:claude-3-5-sonnet-latest"

    research_agent = Agent(
        model_name,
        tools=[
            Tool(visit_webpage_tool, takes_ctx=True),
            Tool(
                DuckDuckGoSearchTool(DDGS()).__call__,
                takes_ctx=False,
                prepare=select_search,
            ),
            Tool(hacker_news_tool, takes_ctx=True, prepare=select_hn),
        ],
        output_type=ResearchResult,
        deps_type=RunDeps,
        retries=4,
        system_prompt=(
            "Your task is to research information to show in a presentation for a boutique data consultancy."
            "The consultants are especially interested in topics on Data Engineering, new promising Tools (preferrably in Python), MLOps or AI developments."
            "Please provide the main information you collect verbatim in plain text (you can remove artifacts from websites), and all relevant links and images you find"
        ),
    )

    @research_agent.system_prompt
    def add_research_option(ctx: RunContext[RunDeps]) -> str:
        if ctx.deps.search_goal == "HN":
            return "Please search in the trending stories in Hacker News for a promising topic"
        else:
            return f"Please search the web for more information on the topic {ctx.deps.search_goal} to prepare the presentation."

    research_result = asyncio.run(get_research_result(research_agent))

    # create final Markdown document
    result_md = research_result.full_text + "\n\n"
    for _img in research_result.images:
        result_md += f"![{_img.description}]({_img.url})\n"
    result_md += "\n## References:\n\n"
    for _ref in research_result.references:
        result_md += f"- [{_ref.description}]({_ref.url})\n"

    outdir = os.environ["OUTPUT_DIR"]
    if outdir == "":
        outdir = "./"
    outfile = Path(outdir) / f"markdown_report_{datetime.now()}.md"
    outfile.write_text(result_md)
