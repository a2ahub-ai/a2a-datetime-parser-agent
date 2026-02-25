# A2A Datetime Parser Agent

An autonomous agent that provides datetime parsing capabilities using the A2A SDK.

## Features

- **Datetime Parsing**: Parse natural language date and time expressions.
- **Natural Language Query**: Ask generic questions about dates (e.g., "What is the date for next Friday?").
- **A2A Protocol**: Fully compliant with the Agent-to-Agent protocol.

## Prerequisites

- Python 3.12+
- `uv` package manager (recommended)
- OpenAI or Groq API Key

## Setup

1.  **Install dependencies**:

    ```bash
    uv sync
    ```

2.  **Environment Configuration**:

    Create a `.env` file from the example:

    ```bash
    cp .env.example .env
    ```

    Edit `.env` and provide your API keys:
    - `OPENAI_API_KEY` or `GROQ_API_KEY`: Required for the LLM.
    - `ACCUWEATHER_API_KEY`: Required for weather data.

## Running the Agent Server

Start the datetime parser agent server:

```bash
uv run __main__.py
```

The agent will be available at `http://localhost:10001` (or the port specified in your `.env` file).

## Running the CLI Client

You can use the provided CLI tool to interact with the agent for testing.

```bash
uv run cli --agent "http://localhost:10001"
```

### CLI Usage

Once the CLI is running, you can ask datetime-related questions directly.

Examples:
- "What is the date for next Friday?"
- "Forecast for New York tomorrow"
- "Is it raining in London?"

To exit the CLI, type `:q` or `quit`.

## Development

- **Project Structure**:
    - `app/`: Contains the server and agent logic.
        - `server_mcp.py`: The FastMCP server handling tool execution.
        - `server_agent.py`: The MCP client and agent orchestration.
    - `cli/`: The command-line interface client.

- **Dependencies**: Managed via `pyproject.toml`.

