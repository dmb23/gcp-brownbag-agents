from datetime import datetime
from pathlib import Path
from typing import List, Optional

import httpx
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from gcp_brownbag_agents import prompts, types
from gcp_brownbag_agents.tools import (
    DuckDuckGoSearchTool,
    HackerNewsTool,
    WebpageTool,
)


class ConsideredTopic(BaseModel):
    """A possible topic, evaluated on its relevance for data professionals."""

    topic: str
    description: str
    source_url: str  # Where the topic was found
    relevance_score: float  # 0-1 score of relevance to data engineering/ML


class TopicSelectionResult(BaseModel):
    """Result of the topic selection process."""

    selected_topic: ConsideredTopic
    considered_topics: List[str]


class EnhancedResearchResult(BaseModel):
    """Enhanced research result with additional information."""

    topic: str
    original_description: str
    original_source: str
    technical_details: List[str]
    business_impact: str
    drawbacks: List[str]
    key_insights: List[str]
    code_examples: List[str]
    references: list[types.ReferenceLink]
    images: list[types.ImageLink]


class GrimaudAgent:
    """
    A class that handles all aspects of the Grimaud research agent.
    This includes topic selection, research, and report generation.
    """

    def __init__(
        self,
        model: Model | str,
        request_limit: int = 10,
        retries: int = 2,
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
        self.retries = retries
        self._client_kwargs = {}
        self._client = httpx.AsyncClient(**self._client_kwargs)
        self._usage = UsageLimits(request_limit=self.request_limit)

        # Create tool instances that will be shared across agents
        self.hn_tool = HackerNewsTool()
        self.ddg_tool = DuckDuckGoSearchTool()
        self.webpage_tool = WebpageTool()

        # Initialize the specialized agents
        self.topic_selector = self._create_topic_selector()
        self.researcher = self._create_researcher()
        self.report_generator = self._create_report_generator()

    def _start_client(self):
        if not hasattr(self._client, "is_closed") or self._client.is_closed:
            self._client = httpx.AsyncClient(**self._client_kwargs)

    async def _close_client(self):
        if not self._client.is_closed:
            await self._client.aclose()

    def _create_topic_selector(self) -> Agent[types.RunDeps, TopicSelectionResult]:
        """Create an agent specialized in selecting interesting topics."""
        topic_selector = Agent(
            self.model,
            tools=[self.hn_tool.get_tool()],
            output_type=TopicSelectionResult,
            deps_type=types.RunDeps,
            retries=self.retries,
            system_prompt=prompts.TOPIC_SELECTOR_SYSTEM,
            instrument=True,
        )
        return topic_selector

    def _create_researcher(self) -> Agent[types.RunDeps, EnhancedResearchResult]:
        """Create an agent specialized in in-depth research."""
        researcher = Agent(
            self.model,
            tools=[
                self.webpage_tool.get_tool(),
                self.ddg_tool.get_tool(),
            ],
            output_type=EnhancedResearchResult,
            deps_type=types.RunDeps,
            retries=self.retries,
            system_prompt=prompts.RESEARCHER_SYSTEM,
            instrument=True,
        )
        return researcher

    def _create_report_generator(self) -> Agent:
        """Create an agent specialized in creating polished reports."""
        report_generator = Agent(
            self.model,
            tools=[],
            output_type=str,
            retries=self.retries,
            system_prompt=prompts.REPORT_GENERATOR_SYSTEM,
            instrument=True,
        )
        return report_generator

    async def select_topic(self) -> TopicSelectionResult:
        """
        Step 1: Select an interesting topic from trending sources.

        Returns:
            A selected topic with relevance score and reasoning
        """
        usage_limits = UsageLimits(request_limit=self.request_limit // 3)

        research_deps = types.RunDeps(client=self._client)
        run_result = await self.topic_selector.run(
            prompts.TOPIC_SELECTOR_TASK,
            deps=research_deps,
            usage_limits=usage_limits,
        )

        return run_result.output

    async def research_topic(
        self, topic: TopicSelectionResult
    ) -> EnhancedResearchResult:
        """
        Step 2: Conduct in-depth research on the selected topic.

        Args:
            topic: The selected topic to research

        Returns:
            Comprehensive research results
        """
        # Format the research task with the topic and plan details
        research_task = prompts.RESEARCHER_TASK_TEMPLATE.format(
            topic=topic.selected_topic.topic,
            description=topic.selected_topic.description,
            source_url=topic.selected_topic.source_url,
        )

        research_deps = types.RunDeps(client=self._client)
        run_result = await self.researcher.run(
            research_task, deps=research_deps, usage_limits=self._usage
        )

        return run_result.output

    async def generate_report(self, research_result: EnhancedResearchResult) -> str:
        """
        Step 3: Generate a polished markdown report from research results.

        Args:
            research_result: The comprehensive research results

        Returns:
            Formatted markdown report
        """
        run_result = await self.report_generator.run(
            prompts.REPORT_GENERATOR_TASK_TEMPLATE.format(
                research_result=research_result.model_dump_json()
            ),
            usage_limits=self._usage,
        )

        return run_result.output

    def save_output(self, file_contents: str, filename: Optional[str] = None) -> Path:
        """
        Save (intermediate) output to a file.

        Args:
            file_contents: The string content to save
            filename: Optional custom filename, defaults to timestamp-based name for markdown report

        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = f"markdown_report_{datetime.now()}.md"

        outdir = Path(self.output_dir)
        outdir.mkdir(exist_ok=True, parents=True)

        outfile = outdir / filename
        outfile.write_text(file_contents)

        return outfile

    async def run_full_workflow(self) -> Path:
        """
        Run the complete three-step workflow: topic selection, research, and report generation.

        Returns:
            Path to the saved markdown file
        """
        self._start_client()
        now = datetime.now()
        # Step 1: Select a topic
        topic = await self.select_topic()
        topic_name = topic.selected_topic.topic
        # TODO: change print statements to logging, ideally loguru + logfire instrumentation
        print(
            f"Selected topic: {topic.selected_topic.topic} (Relevance: {topic.selected_topic.relevance_score})"
        )
        print(f"Description: {topic.selected_topic.description}")
        self.save_output(
            topic.model_dump_json(),
            f"topic_selection_{topic_name[:10].replace(' ', '_')}_{now}.json",
        )

        # Step 2: Research the topic
        research_result = await self.research_topic(topic)
        print(
            f"Research completed with {len(research_result.key_insights)} key insights"
        )
        self.save_output(
            research_result.model_dump_json(),
            f"research_result_{topic_name[:10].replace(' ', '_')}_{now}.json",
        )

        # Step 3: Generate a report
        markdown_report = await self.generate_report(research_result)
        print(f"Report generated with {len(markdown_report)} characters")

        # Save the final report
        report_file = self.save_output(
            markdown_report,
            f"markdown_report_{topic_name[:10].replace(' ', '_')}_{now}.md",
        )

        await self._close_client()
        return report_file
