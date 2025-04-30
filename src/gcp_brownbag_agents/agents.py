from pydantic_ai import Agent, RunContext

from gcp_brownbag_agents import prompts, types
from gcp_brownbag_agents.tools import (
    HackerNewsTool, 
    DuckDuckGoSearchTool, 
    WebpageTool,
    select_hn,
    select_search
)


def wake_up_grimaud(model_name: str) -> Agent[types.RunDeps, types.ResearchResult]:
    # Create tool instances
    hn_tool = HackerNewsTool(prepare_func=select_hn)
    ddg_tool = DuckDuckGoSearchTool(prepare_func=select_search)
    webpage_tool = WebpageTool()
    
    grimaud = Agent(
        model_name,
        tools=[
            webpage_tool.get_tool(),
            hn_tool.get_tool(),
            ddg_tool.get_tool(),
        ],
        output_type=types.ResearchResult,
        deps_type=types.RunDeps,
        retries=4,
        system_prompt=prompts.GRIMAUD_SYSTEM,
    )

    @grimaud.system_prompt
    def add_research_option(ctx: RunContext[types.RunDeps]) -> str:
        if ctx.deps.search_goal == "HN":
            return "Please search in the trending stories in Hacker News for a promising topic"
        else:
            return f"Please search the web for more information on the topic {ctx.deps.search_goal} to prepare the presentation."

    return grimaud
