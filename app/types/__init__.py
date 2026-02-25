from typing import Any, Optional, TypedDict, Required, NotRequired

from app.constants import ChatCompletionTypeEnum


class ChatCompletionStreamResponseType(TypedDict):
    type: ChatCompletionTypeEnum
    data: Required[Optional[Any]]
    input_tokens: NotRequired[Optional[int]]
    output_tokens: NotRequired[Optional[int]]


class FunctionCallingResponseType(TypedDict):
    name: str
    index: int
    id: str
    arguments: str
