import asyncio
import os

import logfire
from dotenv import load_dotenv
from pydantic_ai.models.gemini import GeminiModel

from gcp_brownbag_agents.agents import GrimaudAgent

if __name__ == "__main__":
    load_dotenv()

    logfire.configure()
    logfire.instrument_httpx(capture_all=True)

    # Claude 3.5 - current setup was developed here, quickly picks a topic and then writes a happy report
    # model_name = "anthropic:claude-3-5-sonnet-latest"

    # Gemini 2.5 Flash - setup worked, but there were inconsistencies in the reports
    # model = GeminiModel("gemini-2.5-flash-preview-04-17", provider="google-vertex")
    model = GeminiModel("gemini-2.5-pro-preview-03-25", provider="google-vertex")

    # Gemini 2.0 Flash - got lost in an eternal research spree...
    # model = GeminiModel(
    #     "gemini-2.0-flash", provider=GoogleVertexProvider(region="europe-west9")
    # )

    # Create the Grimaud agent with our model and settings
    grimaud_agent = GrimaudAgent(
        model=model, 
        request_limit=30,  # Increased limit for the three-step process
        output_dir=os.environ.get("OUTPUT_DIR", "./")
    )

    # Run the complete three-step workflow
    output_file = asyncio.run(grimaud_agent.run_full_workflow())
    print(f"Research workflow completed and saved to: {output_file}")
