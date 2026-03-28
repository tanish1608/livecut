from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import datetime, timedelta
from typing import Any

from .gemini_bridge import GeminiLiveBridge
from .signal_loops import chat_batch_loop, fake_audio_loop, fake_vision_loop, segment_timer_loop
from .tools import ToolRegistry
from .types import StreamSignal

logger = logging.getLogger(__name__)


class LiveCutRuntime:
    """Coordinates multimodal signals and maps them into tool calls."""

    def __init__(
        self,
        tools: ToolRegistry,
        host_username: str,
        segment_minutes: int,
        chat_seconds: int,
        scene_gameplay_focus: str,
        scene_chatting_focus: str,
        input_host_mic: str,
        source_sfx_airhorn: str,
        source_host_prompt_text: str,
        gemini_bridge: GeminiLiveBridge | None = None,
    ) -> None:
        self.tools = tools
        self.host_username = host_username
        self.segment_minutes = segment_minutes
        self.chat_seconds = chat_seconds
        self.scene_gameplay_focus = scene_gameplay_focus
        self.scene_chatting_focus = scene_chatting_focus
        self.input_host_mic = input_host_mic
        self.source_sfx_airhorn = source_sfx_airhorn
        self.source_host_prompt_text = source_host_prompt_text
        self.gemini_bridge = gemini_bridge
        self._queue: asyncio.Queue[StreamSignal] = asyncio.Queue(maxsize=512)
        self._tasks: list[asyncio.Task] = []
        self._host_speaking = False
        self._last_host_speech_ts: datetime | None = None

    async def start(self) -> None:
        producers = [
            self._consume_generator(segment_timer_loop(self.segment_minutes)),
            self._consume_generator(chat_batch_loop(self.chat_seconds)),
        ]

        if self.gemini_bridge is None:
            producers.extend(
                [
                    self._consume_generator(fake_vision_loop(self.host_username)),
                    self._consume_generator(fake_audio_loop()),
                ]
            )
        else:
            await self.gemini_bridge.connect()
            producers.append(self._consume_generator(self.gemini_bridge.signals()))
        consumers = [asyncio.create_task(self._event_consumer(), name="event_consumer")]

        self._tasks = [*producers, *consumers]
        logger.info("LiveCut runtime started with %d tasks", len(self._tasks))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with suppress(asyncio.CancelledError):
                await task
        if self.gemini_bridge is not None:
            await self.gemini_bridge.disconnect()
        logger.info("LiveCut runtime stopped")

    def _consume_generator(self, generator: AsyncIterator[StreamSignal]) -> asyncio.Task:
        async def _run() -> None:
            async for signal in generator:
                await self._queue.put(signal)

        return asyncio.create_task(_run(), name=f"producer:{generator.__class__.__name__}")

    async def _event_consumer(self) -> None:
        while True:
            signal = await self._queue.get()
            try:
                await self._dispatch(signal)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to dispatch signal: %s", signal)

    async def _dispatch(self, signal: StreamSignal) -> None:
        if signal.source == "audio" and signal.kind == "transient_spike":
            if self._is_host_currently_speaking():
                logger.info("Ignoring transient spike while host is speaking")
                return
            await self.tools.execute("momentary_mute", {"input_name": self.input_host_mic})
            return

        if signal.source == "audio" and signal.kind in {"speech", "speech_start", "speech_end"}:
            if signal.kind == "speech_end":
                self._host_speaking = False
            else:
                self._host_speaking = True
                self._last_host_speech_ts = signal.ts
            return

        if signal.source == "vision" and signal.kind == "frame_analysis":
            state = signal.payload.get("state")
            if state == "combat":
                await self.tools.execute("switch_scene", {"scene_name": self.scene_gameplay_focus})
            elif state == "menu":
                await self.tools.execute("switch_scene", {"scene_name": self.scene_chatting_focus})

            killfeed = str(signal.payload.get("killfeed", ""))
            if self.host_username.lower() in killfeed.lower():
                await self.tools.execute("play_sfx", {"source_name": self.source_sfx_airhorn})
            return

        if signal.source == "timer" and signal.kind == "segment_timeout":
            minutes = signal.payload.get("minutes", self.segment_minutes)
            text = f"You've been on this topic for {minutes} minutes. Time for Q&A."  # host prompt hook
            await self.tools.execute("show_lower_third", {"text": text, "source_name": self.source_host_prompt_text})
            if self.gemini_bridge is not None:
                await self.gemini_bridge.inject_system_message(text)
            return

        if signal.source == "chat" and signal.kind == "chat_batch":
            messages = signal.payload.get("messages", [])
            best_question = self._pick_question(messages)
            if best_question:
                await self.tools.execute("highlight_question", {"question": best_question})
            return

        if signal.source == "gemini" and signal.kind == "tool_call":
            await self._execute_gemini_tool_call(signal.payload)

    async def _execute_gemini_tool_call(self, payload: dict[str, Any]) -> None:
        tool_name = payload.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            logger.warning("Discarding gemini tool call without a valid name: %s", payload)
            return

        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}

        call_id = payload.get("id")
        if call_id is not None and not isinstance(call_id, str):
            call_id = None

        result: dict[str, Any]
        try:
            result = await self.tools.execute(tool_name, arguments)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gemini tool call failed: %s", tool_name)
            result = {"ok": False, "error": str(exc), "tool": tool_name}

        if self.gemini_bridge is not None and call_id:
            await self.gemini_bridge.send_tool_result(call_id=call_id, tool_name=tool_name, result=result)

    def _is_host_currently_speaking(self) -> bool:
        if self._host_speaking:
            return True
        if self._last_host_speech_ts is None:
            return False
        return (datetime.utcnow() - self._last_host_speech_ts) <= timedelta(milliseconds=850)

    @staticmethod
    def _pick_question(messages: list[str]) -> str | None:
        candidates = [m.strip() for m in messages if isinstance(m, str) and "?" in m]
        if not candidates:
            return None
        # Simple baseline ranking: choose the longest question as proxy for quality.
        return max(candidates, key=len)
