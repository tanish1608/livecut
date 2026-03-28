from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class StreamSignal:
    source: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=datetime.utcnow)
