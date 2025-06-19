"""
Example demonstrating how to use mcp_use with Airbnb.

This example shows how to connect an LLM to Airbnb through MCP tools
to perform tasks like searching for accommodations.

Special Thanks to https://github.com/openbnb-org/mcp-server-airbnb for the server.
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from mcp_use import MCPAgent, MCPClient


async def run_airbnb_example():
    """Run an example using Airbnb MCP server."""
    # Load environment variables
    load_dotenv()

    # Create MCPClient with Airbnb configuration
    client = MCPClient.from_config_file(os.path.join(os.path.dirname(__file__), "airbnb_mcp.json"))

    # Create LLM - you can choose between different models
    llm = ChatOllama(model="qwen3:30b", base_url="http://localhost:11434", temperature=-0.1)

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=10, verbose=True)

    try:
        # Run a query to search for accommodations
        result = await agent.run(
            "Find me a nice place to stay in Göteborg, Sweden for 2 adults "
            "for a week in August 2025. I prefer places with a pool and "
            "good reviews. Show me the top 3 options.",
            max_steps=30,

        )
        print(f"\nResult: {result}")
    finally:
        # Ensure we clean up resources properly
        if client.sessions:
            await client.close_all_sessions()


if __name__ == "__main__":
    asyncio.run(run_airbnb_example())
