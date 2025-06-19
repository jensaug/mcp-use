"""
This example shows how to test the different functionalities of MCPs using the MCP server from
anthropic.
"""

import asyncio

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from mcp_use import MCPAgent, MCPClient

everything_server = {
    "mcpServers": {"everything": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-everything"]}}
}


async def main():
    """Run the example using a configuration file."""
    load_dotenv()
    client = MCPClient(config=everything_server)
    llm = ChatOllama(model="qwen3:30b", base_url="http://localhost:11434", temperature=-0.1)
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    result = await agent.run(
        """
        Hello, you are a tester can you please answer the follwing questions:
        - Which resources do you have access to?
        - Which prompts do you have access to?
        - Which tools do you have access to?
        """,
        max_steps=30,
    )
    print(f"\nResult: {result}")


if __name__ == "__main__":
    # Run the appropriate example
    asyncio.run(main())
