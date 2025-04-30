import asyncio
import os
from datetime import datetime
from pathlib import Path

import httpx
import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent
from tenacity import retry, stop_after_attempt, wait_exponential

from gcp_brownbag_agents import agents, prompts, types


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4))
async def let_grimaud_search(
    research_agent: Agent[types.RunDeps, types.ResearchResult],
) -> types.ResearchResult:
    # can raise a status code 529 "Overloaded"!
    async with httpx.AsyncClient() as _client:
        research_deps = types.RunDeps(client=_client, search_goal="HN")
        run_result = await research_agent.run(prompts.GRIMAUD_TASK, deps=research_deps)

    return run_result.output


if __name__ == "__main__":
    load_dotenv()

    logfire.configure()
    logfire.instrument_httpx(capture_all=True)

    model_name = "anthropic:claude-3-5-sonnet-latest"

    grimaud = agents.wake_up_grimaud(model_name)
    research_result = asyncio.run(let_grimaud_search(grimaud))

    # create final Markdown document
    result_md = research_result.full_text + "\n\n"
    for _img in research_result.images:
        result_md += f"![{_img.description}]({_img.url})\n"
    result_md += "\n## References:\n\n"
    for _ref in research_result.references:
        result_md += f"- [{_ref.description}]({_ref.url})\n"

    outdir = os.environ.get("OUTPUT_DIR", "./")
    outfile = Path(outdir) / f"markdown_report_{datetime.now()}.md"
    outfile.write_text(result_md)
