
import json
from contextlib import AsyncExitStack
from typing import AsyncGenerator, List, Dict, Any, cast
from openai.types import ResponseFormatJSONSchema
from openai.types.shared.response_format_json_schema import JSONSchema

import httpx
from mcp import ClientSession
from mcp.client.stdio import (  # For JSON-RPC stdio transport
    StdioServerParameters,
    stdio_client,
)
from mcp.client.streamable_http import streamablehttp_client  # For HTTP transport
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam

from app.config.settings import BaseConfig
from app.constants import ChatCompletionTypeEnum, AGENT_DESCRIPTION
from app.lib.llm.groq import GroqLLMProvider
from app.lib.llm.ollama import OllamaLLMProvider
from app.lib.llm.openai import OpenAILLMProvider
from app.types import ChatCompletionStreamResponseType
from app.utils.logger import logger
from app.server_mcp import DatetimeParserTool


class LoggingHTTPClient(httpx.AsyncClient):
    """Custom HTTP client that logs all requests"""

    async def send(self, request, **kwargs):
        logger.info(f"🌐 HTTP REQUEST to {request.url}")
        logger.info(f"Method: {request.method}")
        logger.info(f"Headers: {dict(request.headers)}")

        if request.content:
            try:
                # Try to parse and pretty-print JSON content
                content = json.loads(request.content.decode())
                logger.info("📤 Request Body:")
                logger.info(json.dumps(content, indent=2))

                # Specifically highlight messages and tools
                if "messages" in content:
                    logger.info("💬 MESSAGES TO LLM:")
                    for i, msg in enumerate(content["messages"]):
                        logger.info(f"Message {i + 1}: {json.dumps(msg, indent=2)}")

                if "tools" in content:
                    logger.info("🛠️  TOOLS SCHEMA:")
                    logger.info(json.dumps(content["tools"], indent=2))

            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning(f"Request Body (raw): {request.content}")

        logger.info("-" * 80)

        response = await super().send(request, **kwargs)

        logger.success(f"✅ HTTP RESPONSE from {request.url}")
        logger.info(f"Status: {response.status_code}")
        if response.content:
            try:
                response_content = json.loads(response.content.decode())
                logger.info("📥 Response Body:")
                logger.info(json.dumps(response_content, indent=2))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning(f"Response Body (raw): {response.content}")
        logger.info("=" * 80)

        return response


class AgentServer:
    def __init__(self):
        # Initialize session and client objects for multiple servers
        self.servers: dict[str, ClientSession] = {}  # Map server names to sessions
        self.exit_stack = AsyncExitStack()

        # Create custom HTTP client for logging
        self.http_client = LoggingHTTPClient()

        self.llm = GroqLLMProvider(api_key=BaseConfig.GROQ_API_KEY, model_name="openai/gpt-oss-120b")
        # self.llm = OpenAILLMProvider(api_key=BaseConfig.OPENAI_API_KEY, model_name="gpt-4.1-nano")
        # self.llm = OllamaLLMProvider(api_key="", model_name="lfm2.5-thinking:latest") # phi4-mini:latest qwen3:4b gemma3:4b-it-qat qwen3:1.7b lfm2.5-thinking:latest

    async def connect_to_server(self, server_name: str, url: str):
        """Connect to an MCP server over HTTP

        Args:
            server_name: A unique name for this server connection
            url: The HTTP endpoint URL of the running MCP server (e.g., "http://127.0.0.1:5000/mcp")
        """
        logger.info(f"🔌 Connecting to MCP server '{server_name}' at: {url}")

        # Connect using Streamable HTTP transport
        http_transport = await self.exit_stack.enter_async_context(
            streamablehttp_client(url)
        )
        read, write, _ = http_transport
        session = await self.exit_stack.enter_async_context(ClientSession(read, write))

        await session.initialize()

        # List available tools
        response = await session.list_tools()
        tools = response.tools
        logger.success(
            f"✅ Connected to server '{server_name}' with tools: {[tool.name for tool in tools]}"
        )
        print(
            f"\nConnected to server '{server_name}' with tools:",
            [tool.name for tool in tools],
        )

        # Store the session
        self.servers[server_name] = session

    async def connect_to_stdio_server(self, server_name: str, command: list[str]):
        """Connect to an MCP server over JSON-RPC stdio transport

        Args:
            server_name: A unique name for this server connection
            command: Command to start the server (e.g., ["python", "music-agent.py"])
        """
        logger.info(
            f"🔌 Connecting to JSON-RPC MCP server '{server_name}' with command: {' '.join(command)}"
        )

        # Create server parameters with proper structure
        server_params = StdioServerParameters(
            command=command[0],  # First element is the executable
            args=command[1:] if len(command) > 1 else [],  # Rest are arguments
        )

        # Connect using stdio transport
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(read, write))

        await session.initialize()

        # List available tools
        response = await session.list_tools()
        tools = response.tools
        logger.success(
            f"✅ Connected to JSON-RPC server '{server_name}' with tools: {[tool.name for tool in tools]}"
        )
        print(
            f"\nConnected to JSON-RPC server '{server_name}' with tools:",
            [tool.name for tool in tools],
        )

        # Store the session
        self.servers[server_name] = session

    async def process_query(self, messages: List[ChatCompletionMessageParam], 
                            extra_arguments: Dict[str, Any] = None
                            ) -> AsyncGenerator[ChatCompletionStreamResponseType, None]:
        """Process a query using GroqLLMProvider and available tools"""
        logger.info("🚀 Processing new query")

        tool_name = DatetimeParserTool().name
        instruction = f"[VERY IMPORTANT] Remember that you have only one task {tool_name}. Do not use the system time of the model for calculation; For example, last year is -1 year from the current date. You are not allowed to perform any other tasks such as answering user questions. Only return results using ResponseFormatJSONSchema output {tool_name} as JSON format (do not use tool call), do not answer anything else."
        logger.info(f"📝 Instruction: {instruction}")

        # Add AGENT_DESCRIPTION as system message at the beginning
        system_message: ChatCompletionMessageParam = {
            "role": "system",
            "content": instruction
        }
        messages = [system_message] + messages

        logger.info(f"📝 Messages: {messages}")

        response_json_schema = None

        tool_to_server_map = {}  # Map tool names to their server sessions

        for server_name, session in self.servers.items():
            response = await session.list_tools()
            for tool in response.tools:
                response_json_schema = JSONSchema(
                    name=tool.name,
                    description=tool.description,
                    schema=tool.inputSchema,
                )
                tool_to_server_map[tool.name] = (server_name, session)
                break

        # Initial GroqLLMProvider call
        logger.info("📞 Making initial GroqLLMProvider call...")
        function_calls = []

        if response_json_schema is not None:
            async for response_chunk in self.llm.chat_completion(  # type: ignore
                messages=messages,
                # response_format=ResponseFormatJSONSchema(type="json_schema", json_schema=response_json_schema),
                response_format=cast(ResponseFormatJSONSchema, {
                    "type": "json_schema",
                    "json_schema": {
                            "name": response_json_schema.name,
                            "description": response_json_schema.description,
                            "schema": response_json_schema.schema_,
                    }
                }),
                temperature=1,
                # reasoning_effort="low"
            ):
                logger.debug(f"Response chunk: {response_chunk}")
                if response_chunk["type"] == ChatCompletionTypeEnum.CONTENT:
                    yield response_chunk
                elif response_chunk["type"] == ChatCompletionTypeEnum.FUNCTION_CALLING:
                    if response_chunk.get("data") and isinstance(
                            response_chunk["data"], dict) and response_chunk["data"].get("function"):
                        function_calls = response_chunk["data"]["function"]
                        logger.info(f"🔧 LLM requested {len(function_calls)} tool call(s)")

                elif response_chunk["type"] == ChatCompletionTypeEnum.DONE:
                    yield ChatCompletionStreamResponseType(
                        type=ChatCompletionTypeEnum.DONE,
                        data=None)
                    break

            # Process tool calls if any
            if function_calls:
                # Add assistant message with tool calls to conversation
                tool_calls = []

                for func_call in function_calls:
                    tool_calls.append({
                        "id": func_call.get("id", f"call_{func_call['name']}"),
                        "type": "function",
                        "function": {
                            "name": func_call["name"],
                            "arguments": str(func_call["arguments"]) if isinstance(func_call["arguments"], dict) else func_call["arguments"]
                        }
                    })

                assistant_message: ChatCompletionMessageParam = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls
                }
                messages.append(assistant_message)

                tool_results = {}
                for func_call in function_calls:
                    tool_name = response_json_schema.name
                    tool_args = func_call["arguments"]

                    # Ensure tool_args is a dict if it comes as a JSON string
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse tool arguments: {tool_args}")
                            tool_args = {}

                    if extra_arguments and isinstance(tool_args, dict):
                        logger.info(f"Injecting extra arguments into tool call: {extra_arguments}")
                        tool_args.update(extra_arguments)

                    # Find which server has this tool
                    if tool_name in tool_to_server_map:
                        server_name, session = tool_to_server_map[tool_name]
                        logger.info(
                            f"⚙️  Executing tool: {tool_name} on server '{server_name}' with args: {tool_args}")

                        # Execute tool call on the appropriate server
                        result = await session.call_tool(tool_name, tool_args)
                        logger.info(
                            f"✅ Tool result from '{server_name}': {result.content}"
                        )
                        tool_results[tool_name] = result
                        yield ChatCompletionStreamResponseType(
                            type=ChatCompletionTypeEnum.DATA,
                            data=tool_results)
                    else:
                        logger.error(
                            f"❌ Tool {tool_name} not found in any connected server"
                        )
                        tool_results[tool_name] = f"Error: Tool {tool_name} not available"
                        yield ChatCompletionStreamResponseType(
                            type=ChatCompletionTypeEnum.DATA,
                            data=tool_results)
                        
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
        await self.http_client.aclose()

