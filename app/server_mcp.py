from asyncio import sleep
import os
import sys
from typing import Any, Dict
from datetime import datetime

from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent

from app.config.settings import BaseConfig
from app.utils.datetime import (
    convert_datetime_payload,
    TimeInputPayload,
    TimeSingle,
    TimeRange,
    TimeRangeDate,
    AbsoluteTime,
    RelativeTime
)

# Initialize FastMCP server
mcp = FastMCP(f"{BaseConfig.SERVICE_NAME}-mcp-server")


class DatetimeParserTool(Tool):
    name: str = "datetime_parser"
    description: str = f"Extract datetime information from user's command into structured datetime formats."
    parameters: Dict[str, Any] = {
        "type": "object",
        "description": "absolute: the absolute date and time, E.g: '2023-07-30' => {start: {mode: 'absolute', year: 2023, month: 7, day: 30}}, July 30th at 3 PM => {start: {mode: 'absolute', month: 7, day: 30, hour: 15}}\nrelative: yesterday(day -1), today(day 0), tomorrow(day 1), last week (day -7), last month(month -1)\nNested time: E.g: 'one month ago at 3 PM' => {start: {mode: 'relative', month: -1, extended_time: {mode: 'absolute', hour: 15}}}.",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Provide reasoning for the datetime parsing result.",
            },
            "parsable": {
                "type": "boolean",
                "description": "Indicates whether the datetime information could be parsed from the input.",
            },
            "start": {
                "type": "object",
                "description": "Start datetime object",
                "default": {},
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["absolute", "relative"]
                    },
                    "year": {
                        "type": "integer"
                    },
                    "month": {
                        "type": "integer"
                    },
                    "day": {
                        "type": "integer"
                    },
                    "hour": {
                        "type": "integer"
                    },
                    "minute": {
                        "type": "integer"
                    },
                    "extended_time": {
                        "type": "object",
                        "description": "Nested object",
                        "$ref": "#",
                    },
                },
            },
            "end": {
                "type": "object",
                "description": "End datetime object",
                "default": {},
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["absolute", "relative"]
                    },
                    "year": {
                        "type": "integer"
                    },
                    "month": {
                        "type": "integer"
                    },
                    "day": {
                        "type": "integer"
                    },
                    "hour": {
                        "type": "integer"
                    },
                    "minute": {
                        "type": "integer"
                    },
                    "extended_time": {
                        "type": "object",
                        "description": "Nested object",
                        "$ref": "#",
                    },
                },
            },
        }
    }

    async def run(self, arguments: Dict[str, Any]) -> ToolResult:
        # Use sys.stderr for logging to avoid interfering with MCP stdio transport
        import sys
        print(f"[datetime-parser-mcp-server] Received arguments: {arguments}", file=sys.stderr)

        # Extract the start and end datetime objects from arguments
        start_data = arguments.get("start", {})
        end_data = arguments.get("end", {})

        # Build TimeInputPayload based on whether we have start and end
        payload = None

        # Helper function to build absolute/relative time objects
        def build_time_components(data: Dict[str, Any]):
            mode = data.get("mode")
            abs_time = None
            rel_time = None
            now_flag = False

            if mode == "now":
                now_flag = True
            elif mode == "absolute":
                abs_time = AbsoluteTime(
                    year=data.get("year"),
                    month=data.get("month"),
                    day=data.get("day"),
                    hour=data.get("hour"),
                    minute=data.get("minute")
                )
            elif mode == "relative":
                rel_time = RelativeTime(
                    year=data.get("year"),
                    month=data.get("month"),
                    day=data.get("day"),
                    hour=data.get("hour"),
                    minute=data.get("minute")
                )

            return abs_time, rel_time, now_flag

        # Determine if we have a time range or single time
        if start_data and end_data:
            # Time range scenario
            start_abs, start_rel, start_now = build_time_components(start_data)
            end_abs, end_rel, end_now = build_time_components(end_data)

            payload = TimeInputPayload(
                time_range=TimeRange(
                    start_date=TimeRangeDate(
                        absolute=start_abs,
                        relative=start_rel,
                        now=start_now if start_now else None
                    ),
                    end_date=TimeRangeDate(
                        absolute=end_abs,
                        relative=end_rel,
                        now=end_now if end_now else None
                    )
                )
            )
        elif start_data:
            # Single time scenario
            start_abs, start_rel, start_now = build_time_components(start_data)

            payload = TimeInputPayload(
                time_single=TimeSingle(
                    absolute=start_abs,
                    relative=start_rel,
                    now=start_now if start_now else None
                )
            )
        else:
            # No time specified, use empty payload (defaults to current time)
            payload = TimeInputPayload()

        # Get current datetime as ISO string
        current_date_str = datetime.now().isoformat()

        # Convert the payload using the datetime utility
        result = convert_datetime_payload(payload, current_date_str)

        # Format the result for MCP response
        response_data: Dict[str, Any] = {
            "parsable": result.parsable
        }

        if result.reason:
            response_data["reason"] = result.reason

        if result.time_single:
            response_data["time_single"] = result.time_single.model_dump(exclude_none=True)
        elif result.time_range:
            response_data["time_range"] = {
                "start_date": result.time_range["start_date"].model_dump(exclude_none=True),
                "end_date": result.time_range["end_date"].model_dump(exclude_none=True)
            }

        # Use sys.stderr for logging to avoid interfering with MCP stdio transport
        import sys
        print(f"[datetime-parser-mcp-server] Converted result: {response_data}", file=sys.stderr)

        return ToolResult(
            structured_content=response_data
        )


# Add the tool to the server using the correct method
mcp.add_tool(DatetimeParserTool())

if __name__ == "__main__":
    # Use sys.stderr for logging to avoid interfering with MCP stdio transport
    print("Starting MCP Server with stdio transport", file=sys.stderr)
    mcp.run(transport="stdio")
