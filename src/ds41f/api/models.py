"""OpenAI-shaped request/response models (transport-independent).

The HTTP layer and the V4.1 prompt codec are deliberately not part of the engine;
this module defines the stable wire shapes the API layer will serve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class ChatCompletionRequest:
    model: str
    messages: Sequence[dict]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Sequence[str] = ()
    stream: bool = False


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Choice:
    index: int
    finish_reason: str
    message: dict = field(default_factory=dict)


@dataclass
class ChatCompletion:
    id: str
    model: str
    choices: list[Choice]
    usage: Usage
