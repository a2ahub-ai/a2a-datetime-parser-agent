import json
import os
import sys
import uuid
import asyncio
from typing import List, cast, Dict, Any, Optional

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCard,
    TaskState,
    TextPart,
    Part,
    DataPart,
    UnsupportedOperationError,
    Role,
)
from a2a.utils.errors import ServerError
from a2a.utils import new_agent_text_message, new_task

from openai.types.chat import ChatCompletionMessageParam

from .server_agent import AgentServer
from app.utils.logger import logger

from app.constants import ChatCompletionTypeEnum


class DatetimeParserAgentExecutor(AgentExecutor):
    """An AgentExecutor that runs an ADK-based Agent for datetime parsing."""

    def __init__(self, runner: AgentServer, card: AgentCard):
        logger.debug("Initializing DatetimeParserAgentExecutor...")
        self.runner = runner
        self._card = card
        self._active_sessions: set[str] = set()

    def _convert_task_history_to_messages(self, task_history) -> List[ChatCompletionMessageParam]:
        """Convert task history to ChatCompletionMessageParam format"""
        messages: List[ChatCompletionMessageParam] = []

        for message in task_history:
            # Extract text content from message parts
            content_parts = []
            if hasattr(message, 'parts') and message.parts:
                for part in message.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        content_parts.append(part.root.text)

            content = " ".join(content_parts) if content_parts else ""

            # Convert role: agent -> assistant, keep user as user
            if hasattr(message, 'role'):
                if message.role == Role.agent:
                    role = "assistant"
                elif message.role == Role.user:
                    role = "user"
                else:
                    role = "user"  # fallback
            else:
                role = "user"  # fallback

            if content.strip():  # Only add messages with content
                if role == "assistant":
                    messages.append(cast(ChatCompletionMessageParam, {
                        "role": "assistant",
                        "content": content
                    }))
                else:  # user role
                    messages.append(cast(ChatCompletionMessageParam, {
                        "role": "user",
                        "content": content
                    }))

        return messages

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        logger.debug("[datetime-parser-agent] execute entered")
        # dump context for debugging
        if context._params:
            logger.debug(context._params.metadata if context._params.metadata else "No metadata")
            # {'single_time_mode': False}
        logger.debug(context.context_id)
        logger.debug(context.task_id)

        query = context.get_user_input()
        task = context.current_task

        if not task:
            if context.message:
                task = new_task(context.message)
                await event_queue.enqueue_event(task)
            else:
                logger.error("No task available and no message to create task from")
                return

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        # Convert task history to messages
        messages = self._convert_task_history_to_messages(task.history)
        
        # Prepare extra arguments for tool execution from context metadata
        extra_arguments = {}
        if context._params and context._params.metadata:
            if 'single_time_mode' in context._params.metadata:
                extra_arguments['single_time_mode'] = context._params.metadata['single_time_mode']

        if not messages and query:
            messages.append(cast(ChatCompletionMessageParam, {
                "role": "user",
                "content": query
            }))

        async for response in self.runner.process_query(messages, extra_arguments=extra_arguments):
            logger.debug(f"[datetime-parser-agent] response type: {response}")

            if response["type"] == ChatCompletionTypeEnum.CONTENT:
                if response["data"]:
                    await updater.update_status(
                        TaskState.completed,
                        new_agent_text_message(response["data"], task.context_id, task.id)
                    )

            elif response["type"] == ChatCompletionTypeEnum.DATA:
                data = response.get("data", {})
                if not data:
                    continue

                combined_response_text = []

                for tool_name, tool_result in data.items():
                    if tool_result and tool_result.structuredContent:
                        await updater.add_artifact(
                            [Part(root=DataPart(data={tool_name: tool_result.structuredContent}, kind="data", metadata=None))],
                            name=f"{tool_name} Data"
                        )
                    elif tool_result:
                        content_text = ""
                        if hasattr(tool_result, 'content'):
                            content_text = " ".join([part.text for part in tool_result.content if part.type == "text"])
                        if content_text.strip():
                            combined_response_text.append(content_text)
                    else:
                        combined_response_text.append(f"No result from {tool_name}")

                logger.debug(f"[status] {TaskState.completed}")
                final_message = " ".join(combined_response_text)
                await updater.update_status(
                    TaskState.completed,
                    new_agent_text_message(final_message, task.context_id, task.id) if final_message.strip() else None
                )

            elif response["type"] == ChatCompletionTypeEnum.DONE:
                pass

        logger.debug("[datetime-parser-agent] execute exiting")

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        logger.debug("[datetime-parser-agent] cancel entered")
        """Cancel the execution for the given context.

        Currently logs the cancellation attempt as the underlying ADK runner
        doesn't support direct cancellation of ongoing tasks.
        """
        session_id = context.context_id
        if session_id in self._active_sessions:
            logger.info(
                f"Cancellation requested for active datetime-parser-agent session: {session_id}"
            )
            # TODO: Implement proper cancellation when ADK supports it
            self._active_sessions.discard(session_id)
        else:
            logger.debug(
                f"Cancellation requested for inactive datetime-parser-agent session: {session_id}"
            )

        raise ServerError(error=UnsupportedOperationError())

