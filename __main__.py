import asyncio
import sys
import os
import uvicorn
import contextlib

# Force UTF-8 encoding for Windows to handle Vietnamese characters
if sys.platform == 'win32':
    # Set environment variable for subprocesses
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from app.constants import AGENT_DESCRIPTION
from app.server_agent import (
    AgentServer,
)
from app.server_executor import (
    DatetimeParserAgentExecutor,
)

from app.utils.logger import logger
from app.config.settings import BaseConfig

DEFAULT_HOST = BaseConfig.HOST
DEFAULT_PORT = BaseConfig.PORT


async def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    skill = AgentSkill(
        id=BaseConfig.AGENT_ID,
        name="Datetime Parsing Skill",
        description="Parse natural language datetime expressions into structured datetime formats.",
        tags=[
            "datetime parsing",
        ],
        examples=[
            "July 30th",
            "July 30th at 2 AM",
            "From 2 AM on July 30th to 5 AM on July 31st",
            "Two days before last month",
            "Three hours ago",
            "From two hours ago to three hours ago",
            "From two hours after now to 4 PM",
            "Two days ago at 13 PM",
        ])

    agent_card = AgentCard(
        name=BaseConfig.AGENT_NAME,
        description=AGENT_DESCRIPTION,
        url=BaseConfig.APP_URL,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    runner = AgentServer()
    # Use -X utf8 flag to ensure UTF-8 encoding for the subprocess on Windows
    python_cmd = ["python",
                  "-X",
                  "utf8",
                  "app/server_mcp.py"] if sys.platform == 'win32' else ["python",
                                                                        "app/server_mcp.py"]
    await runner.connect_to_stdio_server("datetime-parser-agent", python_cmd)

    agent_executor = DatetimeParserAgentExecutor(runner, agent_card)

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    routes = a2a_app.routes()

    app = Starlette(
        routes=routes,
    )

    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main(DEFAULT_HOST, DEFAULT_PORT))
