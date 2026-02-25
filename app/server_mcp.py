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
    RelativeTime,
    WeekdayOffset,
    WEEKDAY_MAP,
)

# Initialize FastMCP server
mcp = FastMCP(f"{BaseConfig.SERVICE_NAME}-mcp-server")


class DatetimeParserTool(Tool):
    name: str = "datetime_parser"
    description: str = f"Decompose time spans within a sentence into a list of independent atomic time elements."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Think carefully and provide an analysis of why you're using these time parameters to generate the results."
            },
            "parsable": {
                "type": "boolean",
                "description": "Indicates whether the datetime information could be parsed from the input."
            },
            "time_elements": {
                "type": "array",
                "description": "Ordered list of atomic time components (left-to-right order in the sentence). Each object contains exactly one time-unit key that it refers to in the user's command; for example, if the time is mentioned as a day, the offset_unit must be day, if it's an hour, the offset_unit must be hour, and similarly for other units.",
                "items": {
                    "type": "object",
                    "oneOf": [
                        {
                            "properties": {
                                "mode": {
                                    "type": "string",
                                    "enum": ["absolute", "relative"]
                                },
                                "time_range": {
                                    "type": "string",
                                    "description": "Indicates whether this time element represents the start or end of a time range. For time ranges (e.g., 'from 2pm to 4pm'), the start time would have 'time_range': 'start' and the end time would have 'time_range': 'end'. For single time points (e.g., 'tomorrow at 3pm'), this field can be set to 'start' or omitted based on your preference, but for consistency, you can treat single time points as having 'time_range': 'start'.",
                                    "enum": ["start", "end"]
                                },
                                "offset_unit": {
                                    "type": "string",
                                    "enum": ["year", "month", "day", "hour", "minute", "second"]
                                },
                                "offset_value": {
                                    "type": "integer",
                                    "description": "For relative times, the integer offset (e.g., day=0 for 'today', day=1 for 'tomorrow', day=-1 for 'yesterday', month=1 for 'next month', year=-1 for 'last year', hour=-1 for 'last hour', etc.). For absolute times, the concrete value (e.g., month=4 for April)."
                                }
                            },
                            "required": ["mode", "time_range", "offset_unit", "offset_value"],
                            "additionalProperties": False
                        },
                        {
                            "properties": {
                                "time_range": {
                                    "type": "string",
                                    "enum": ["start", "end"]
                                },
                                "offset_unit": {
                                    "type": "string",
                                    "enum": ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
                                },
                                "offset_value": {
                                    "type": "integer",
                                    "description": "For weekdays, use offset_unit for the day and offset_value for the occurrence (e.g., offset_unit='monday', offset_value=2 for 'the Monday after next')."
                                }
                            },
                            "required": ["time_range", "offset_unit", "offset_value"],
                            "additionalProperties": False
                        }
                    ]
                }
            },
            "components_count": {
                "type": "integer",
                "description": "Exact length of the time_elements array"
            }
        },
        "required": ["reasoning", "parsable", "time_elements", "components_count"],
        "additionalProperties": False
    }

    async def run(self, arguments: Dict[str, Any]) -> ToolResult:
        import sys
        print(f"[datetime-parser-mcp-server] Received arguments: {arguments}", file=sys.stderr)

        parsable = arguments.get("parsable", False)
        time_elements = arguments.get("time_elements", [])

        # Early exit if not parsable or no elements
        if not parsable or not time_elements:
            response_data: Dict[str, Any] = {
                "parsable": False,
                "reason": arguments.get("reasoning", "Could not parse datetime from input")
            }
            print(f"[datetime-parser-mcp-server] Not parsable result: {response_data}", file=sys.stderr)
            return ToolResult(structured_content=response_data)

        # ── Partition elements by time_range ──
        DATE_UNITS = {"year", "month", "day"}
        start_elements = []
        end_elements = []

        for elem in time_elements:
            tr = elem.get("time_range", "start")
            if tr == "end":
                end_elements.append(elem)
            else:
                start_elements.append(elem)

        # ── Date inheritance: if end group lacks date-level units, copy from start ──
        def _has_date_units(elements):
            for e in elements:
                unit = e.get("offset_unit", "")
                if unit in DATE_UNITS or unit in WEEKDAY_MAP:
                    return True
            return False

        if end_elements and not _has_date_units(end_elements):
            for e in start_elements:
                unit = e.get("offset_unit", "")
                if unit in DATE_UNITS or unit in WEEKDAY_MAP:
                    inherited = dict(e)
                    inherited["time_range"] = "end"
                    end_elements.append(inherited)

        # ── Build time components from a list of elements ──
        def build_components(elements):
            abs_time = AbsoluteTime()
            rel_time = RelativeTime()
            weekday_offset = None
            has_abs = False
            has_rel = False

            for elem in elements:
                unit = elem.get("offset_unit", "")
                value = elem.get("offset_value", 0)
                mode = elem.get("mode")  # None for weekday elements

                # Weekday element (no mode field)
                if unit in WEEKDAY_MAP:
                    weekday_offset = WeekdayOffset(name=unit, offset=value)
                    continue

                # Regular time element
                if mode == "absolute":
                    has_abs = True
                    if hasattr(abs_time, unit):
                        setattr(abs_time, unit, value)
                elif mode == "relative":
                    has_rel = True
                    if hasattr(rel_time, unit):
                        setattr(rel_time, unit, value)

            return (
                abs_time if has_abs else None,
                rel_time if has_rel else None,
                weekday_offset
            )

        # ── Build payload ──
        if end_elements:
            # Time range scenario
            start_abs, start_rel, start_wd = build_components(start_elements)
            end_abs, end_rel, end_wd = build_components(end_elements)

            payload = TimeInputPayload(
                time_range=TimeRange(
                    start_date=TimeRangeDate(
                        absolute=start_abs,
                        relative=start_rel,
                        weekday=start_wd
                    ),
                    end_date=TimeRangeDate(
                        absolute=end_abs,
                        relative=end_rel,
                        weekday=end_wd
                    )
                )
            )
        elif start_elements:
            # Single time scenario
            start_abs, start_rel, start_wd = build_components(start_elements)

            payload = TimeInputPayload(
                time_single=TimeSingle(
                    absolute=start_abs,
                    relative=start_rel,
                    weekday=start_wd
                )
            )
        else:
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
