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
from .vlm_bridge import NvidiaVLMBridge

logger = logging.getLogger(__name__)


class LiveCutRuntime:
    """Coordinates multimodal signals and maps them into tool calls."""

    def __init__(
        self,
        tools: ToolRegistry,
        host_username: str,
        segment_minutes: int,
        chat_seconds: int,
        ai_only_scene_switching: bool,
        gemini_scene_switch_delay_seconds: float,
        scene_gameplay_focus: str,
        scene_chatting_focus: str,
        input_host_mic: str,
        source_sfx_airhorn: str,
        source_host_prompt_text: str,
        status_log_seconds: int,
        run_simulation_loops_with_gemini: bool,
        gemini_bridge: GeminiLiveBridge | None = None,
        vlm_bridge: NvidiaVLMBridge | None = None,
        vlm_scene_switch_delay_seconds: float | None = None,
    ) -> None:
        self.tools = tools
        self.host_username = host_username
        self.segment_minutes = segment_minutes
        self.chat_seconds = chat_seconds
        self.ai_only_scene_switching = ai_only_scene_switching
        self.gemini_scene_switch_delay_seconds = max(0.0, float(gemini_scene_switch_delay_seconds))
        if vlm_scene_switch_delay_seconds is None:
            self.vlm_scene_switch_delay_seconds = self.gemini_scene_switch_delay_seconds
        else:
            self.vlm_scene_switch_delay_seconds = max(0.0, float(vlm_scene_switch_delay_seconds))
        self.scene_gameplay_focus = scene_gameplay_focus
        self.scene_chatting_focus = scene_chatting_focus
        self.input_host_mic = input_host_mic
        self.source_sfx_airhorn = source_sfx_airhorn
        self.source_host_prompt_text = source_host_prompt_text
        self.status_log_seconds = max(5, int(status_log_seconds))
        self.run_simulation_loops_with_gemini = run_simulation_loops_with_gemini
        self.gemini_bridge = gemini_bridge
        self.vlm_bridge = vlm_bridge
        self._queue: asyncio.Queue[StreamSignal] = asyncio.Queue(maxsize=512)
        self._tasks: list[asyncio.Task] = []
        self._host_speaking = False
        self._last_host_speech_ts: datetime | None = None
        self._processed_signals = 0
        self._last_signal_ts: datetime | None = None
        self._pending_vlm_scene_task: asyncio.Task | None = None

    async def start(self) -> None:
        producers = [
            self._consume_generator(segment_timer_loop(self.segment_minutes)),
            self._consume_generator(chat_batch_loop(self.chat_seconds)),
        ]

        if self.gemini_bridge is None and self.vlm_bridge is None:
            producers.extend(
                [
                    self._consume_generator(fake_vision_loop(self.host_username)),
                    self._consume_generator(fake_audio_loop()),
                ]
            )
        else:
            try:
                if self.gemini_bridge is not None:
                    await self.gemini_bridge.connect()
                    producers.append(self._consume_generator(self.gemini_bridge.signals()))
                    await self._send_startup_context_to_gemini()
                if self.vlm_bridge is not None:
                    await self.vlm_bridge.connect()
                    producers.append(self._consume_generator(self.vlm_bridge.signals()))
            except Exception:
                if self.vlm_bridge is not None:
                    with suppress(Exception):
                        await self.vlm_bridge.disconnect()
                if self.gemini_bridge is not None:
                    with suppress(Exception):
                        await self.gemini_bridge.disconnect()
                raise
            if self.run_simulation_loops_with_gemini:
                producers.extend(
                    [
                        self._consume_generator(fake_vision_loop(self.host_username)),
                        self._consume_generator(fake_audio_loop()),
                    ]
                )
                logger.warning(
                    "Hybrid mode enabled: simulation loops run alongside Gemini live session. "
                    "These loops emit random vision/audio events and can cause random scene switches."
                )
            logger.info(
                "Model bridges enabled | gemini=%s vlm=%s",
                self.gemini_bridge is not None,
                self.vlm_bridge is not None,
            )
        consumers = [asyncio.create_task(self._event_consumer(), name="event_consumer")]
        telemetry = [asyncio.create_task(self._status_loop(), name="runtime_status")]

        self._tasks = [*producers, *consumers, *telemetry]
        if self.ai_only_scene_switching:
            logger.info(
                "AI-only scene switching enabled. Local vision heuristics are disabled; "
                "only model tool calls (Gemini/VLM) can invoke switch_scene."
            )
        else:
            logger.warning("AI-only scene switching is disabled. Local heuristics may switch scenes.")
        logger.info("LiveCut runtime started with %d tasks", len(self._tasks))

    async def _send_startup_context_to_gemini(self) -> None:
        if self.gemini_bridge is None:
            return

        context_text = (
            "LiveCut startup context:\n"
            f"- Host username for killfeed detection: {self.host_username}\n"
            f"- Gameplay focus scene: {self.scene_gameplay_focus}\n"
            f"- Chatting focus scene: {self.scene_chatting_focus}\n"
            f"- Host mic input: {self.input_host_mic}\n"
            f"- Airhorn source: {self.source_sfx_airhorn}\n"
            f"- Host prompt source: {self.source_host_prompt_text}\n"
            "Use only exact source/scene names from this context."
        )
        try:
            await self.gemini_bridge.send_user_text(context_text)
            logger.info("Sent startup control-room context to Gemini")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to send startup context to Gemini")

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with suppress(asyncio.CancelledError):
                await task
        if self.gemini_bridge is not None:
            await self.gemini_bridge.disconnect()
        if self.vlm_bridge is not None:
            await self.vlm_bridge.disconnect()
        if self._pending_vlm_scene_task is not None:
            self._pending_vlm_scene_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._pending_vlm_scene_task
            self._pending_vlm_scene_task = None
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
                self._processed_signals += 1
                self._last_signal_ts = signal.ts
                await self._dispatch(signal)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to dispatch signal: %s", signal)

    async def _status_loop(self) -> None:
        while True:
            await asyncio.sleep(self.status_log_seconds)
            last_seen = "never"
            if self._last_signal_ts is not None:
                age_s = (datetime.utcnow() - self._last_signal_ts).total_seconds()
                last_seen = f"{age_s:.1f}s ago"
            logger.info(
                "Runtime heartbeat | queue=%d/%d processed=%d last_signal=%s gemini=%s vlm=%s",
                self._queue.qsize(),
                self._queue.maxsize,
                self._processed_signals,
                last_seen,
                self.gemini_bridge is not None,
                self.vlm_bridge is not None,
            )

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
            if not self.ai_only_scene_switching:
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
            if self.gemini_bridge is not None and messages:
                prompt = (
                    "You are the stream producer. From this chat batch, pick the single best question and "
                    "call highlight_question(question=...). Chat batch:\n- " + "\n- ".join(str(m) for m in messages)
                )
                try:
                    await self.gemini_bridge.send_user_text(prompt)
                    logger.info("Forwarded chat batch to Gemini for question selection")
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to forward chat batch to Gemini")

            best_question = self._pick_question(messages)
            if best_question:
                await self.tools.execute("highlight_question", {"question": best_question})
            return

        if signal.source in {"gemini", "vlm"} and signal.kind == "tool_call":
            await self._execute_model_tool_call(signal.payload, source=signal.source)
            return

        if signal.source in {"gemini", "vlm"} and signal.kind == "assistant_transcript":
            text = str(signal.payload.get("text", "")).strip()
            if text:
                logger.info("%s assistant transcript: %s", signal.source.upper(), text)
            return

        if signal.source == "gemini" and signal.kind == "setup_complete":
            logger.info("Gemini setup complete acknowledged by runtime")

    async def _execute_model_tool_call(self, payload: dict[str, Any], source: str) -> None:
        tool_name = payload.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            logger.warning("Discarding %s tool call without a valid name: %s", source, payload)
            return

        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}

        call_id = payload.get("id")
        if call_id is not None and not isinstance(call_id, str):
            call_id = None

        result: dict[str, Any]
        try:
            switch_delay_seconds = (
                self.vlm_scene_switch_delay_seconds if source == "vlm" else self.gemini_scene_switch_delay_seconds
            )
            if source == "vlm" and tool_name == "switch_scene" and switch_delay_seconds > 0:
                scene_name = arguments.get("scene_name") if isinstance(arguments, dict) else None
                if isinstance(scene_name, str) and scene_name:
                    self._schedule_delayed_vlm_scene_switch(scene_name, delay_seconds=switch_delay_seconds)
                    logger.info(
                        "Scheduled delayed VLM switch_scene to %s in %.2fs",
                        scene_name,
                        switch_delay_seconds,
                    )
                    return
            if tool_name == "switch_scene" and switch_delay_seconds > 0:
                logger.info(
                    "Applying scene switch delay of %.2fs before switch_scene (source=%s)",
                    switch_delay_seconds,
                    source,
                )
                await asyncio.sleep(switch_delay_seconds)
            logger.info("Executing %s tool call: %s args=%s", source, tool_name, arguments)
            result = await self.tools.execute(tool_name, arguments)
            logger.info("%s tool call completed: %s result=%s", source, tool_name, result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s tool call failed: %s", source, tool_name)
            result = {"ok": False, "error": str(exc), "tool": tool_name}

        if source == "gemini" and self.gemini_bridge is not None and call_id:
            await self.gemini_bridge.send_tool_result(call_id=call_id, tool_name=tool_name, result=result)

    def _schedule_delayed_vlm_scene_switch(self, scene_name: str, delay_seconds: float) -> None:
        if self._pending_vlm_scene_task is not None and not self._pending_vlm_scene_task.done():
            self._pending_vlm_scene_task.cancel()

        async def _run() -> None:
            await asyncio.sleep(delay_seconds)
            logger.info("Executing delayed VLM scene switch to %s", scene_name)
            result = await self.tools.execute("switch_scene", {"scene_name": scene_name})
            logger.info("Delayed VLM scene switch result=%s", result)

        task = asyncio.create_task(_run(), name=f"vlm_switch_scene:{scene_name}")

        def _done(done_task: asyncio.Task) -> None:
            if done_task.cancelled():
                return
            try:
                done_task.result()
            except Exception:  # noqa: BLE001
                logger.exception("Delayed VLM scene switch task failed")

        task.add_done_callback(_done)
        self._pending_vlm_scene_task = task

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
