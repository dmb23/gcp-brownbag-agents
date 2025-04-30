from pydantic_ai import Agent, RunContext

from gcp_brownbag_agents import prompts, tools, types


def wake_up_grimaud(model_name: str) -> Agent[types.RunDeps, types.ResearchResult]:
    grimaud = Agent(
        model_name,
        tools=[
            tools.visit_webpage_tool(),
            tools.hacker_news_tool(),
            tools.duckduckgo_search_tool(),
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
