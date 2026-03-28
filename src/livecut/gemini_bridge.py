from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from .types import StreamSignal

logger = logging.getLogger(__name__)


class GeminiLiveBridge:
    """Adapter boundary for Gemini Multimodal Live API integration.

    This class is intentionally small so you can swap in the exact SDK flow
    you use (Vertex AI or Gemini API key auth) without changing the runtime.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._connected = False

    async def connect(self) -> None:
        # TODO: Wire google-genai live session setup here.
        self._connected = True
        logger.info("Gemini bridge initialized for model=%s", self.model)

    async def disconnect(self) -> None:
        self._connected = False

    async def signals(self) -> AsyncIterator[StreamSignal]:
        """Yield structured stream events from Gemini live outputs.

        Expected outputs include speech activity, scene context switches, tool-call
        intents, and entity extraction events.
        """
        if not self._connected:
            raise RuntimeError("Gemini bridge not connected")

        # Placeholder iterator: replace with SDK receive loop.
        if False:
            yield StreamSignal(source="gemini", kind="noop", payload={})
        return
