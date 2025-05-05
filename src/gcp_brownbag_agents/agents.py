from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from gcp_brownbag_agents import prompts, types
from gcp_brownbag_agents.tools import (
    DuckDuckGoSearchTool,
    HackerNewsTool,
    WebpageTool,
    select_hn,
    select_search,
)


class GrimaudAgent:
    """
    A class that handles all aspects of the Grimaud research agent.
    This includes creation, running tasks, and handling results.
    """

    def __init__(
        self,
        model: Model | str,
        request_limit: int = 10,
        retries: int = 4,
        output_dir: str = "./",
    ):
        """
        Initialize the Grimaud agent with the specified model and settings.

        Args:
            model: The LLM model to use
            request_limit: Maximum number of API requests allowed
            retries: Number of retries for failed requests
            output_dir: Directory to save output files
        """
        self.model = model
        self.request_limit = request_limit
        self.output_dir = output_dir
        self.agent = self._create_agent(retries)

    def _create_agent(self, retries: int) -> Agent[types.RunDeps, types.ResearchResult]:
        """Create and configure the agent with all necessary tools and settings."""
        # Create tool instances
        hn_tool = HackerNewsTool(prepare_func=select_hn)
        ddg_tool = DuckDuckGoSearchTool(prepare_func=select_search)
        webpage_tool = WebpageTool()

        grimaud = Agent(
            self.model,
            tools=[
                webpage_tool.get_tool(),
                hn_tool.get_tool(),
                ddg_tool.get_tool(),
            ],
            output_type=types.ResearchResult,
            deps_type=types.RunDeps,
            retries=retries,
            system_prompt=prompts.GRIMAUD_SYSTEM,
            instrument=True,
        )

        @grimaud.system_prompt
        def add_research_option(ctx: RunContext[types.RunDeps]) -> str:
            if ctx.deps.search_goal == "HN":
                return "Please search in the trending stories in Hacker News for a promising topic"
            else:
                return f"Please search the web for more information on the topic {ctx.deps.search_goal} to prepare the presentation."

        return grimaud

    async def run_research(self, search_goal: str = "HN") -> types.ResearchResult:
        """
        Run the research task with the specified search goal.

        Args:
            search_goal: The research goal, default is "HN" for Hacker News

        Returns:
            The research result
        """
        usage_limits = UsageLimits(request_limit=self.request_limit)

        async with httpx.AsyncClient() as client:
            research_deps = types.RunDeps(client=client, search_goal=search_goal)
            run_result = await self.agent.run(
                prompts.GRIMAUD_TASK, deps=research_deps, usage_limits=usage_limits
            )

        return run_result.output

    def convert_to_markdown(self, research_result: types.ResearchResult) -> str:
        """Convert the research result to markdown format."""
        result_md = research_result.full_text + "\n\n"

        for img in research_result.images:
            result_md += f"![{img.description}]({img.url})\n"

        result_md += "\n## References:\n\n"

        for ref in research_result.references:
            result_md += f"- [{ref.description}]({ref.url})\n"

        return result_md

    def save_markdown(
        self, markdown_content: str, filename: Optional[str] = None
    ) -> Path:
        """
        Save the markdown content to a file.

        Args:
            markdown_content: The markdown content to save
            filename: Optional custom filename, defaults to timestamp-based name

        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = f"markdown_report_{datetime.now()}.md"

        outdir = Path(self.output_dir)
        outdir.mkdir(exist_ok=True, parents=True)

        outfile = outdir / filename
        outfile.write_text(markdown_content)

        return outfile

    async def research_and_save(self, search_goal: str = "HN") -> Path:
        """
        Run the complete research workflow: research, convert to markdown, and save.

        Args:
            search_goal: The research goal

        Returns:
            Path to the saved markdown file
        """
        research_result = await self.run_research(search_goal)
        markdown_content = self.convert_to_markdown(research_result)
        return self.save_markdown(markdown_content)


def wake_up_grimaud(
    model: Model | str,
) -> Agent[types.RunDeps, types.ResearchResult]:
    """
    Legacy function to maintain backward compatibility.
    Creates a basic Grimaud agent without the additional functionality.

    Args:
        model: The LLM model to use

    Returns:
        A configured Grimaud agent
    """
    return GrimaudAgent(model)._create_agent(retries=4)
