from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable

import httpx

from .obs_controller import OBSController

logger = logging.getLogger(__name__)
ToolFn = Callable[[dict], Awaitable[dict]]


class ToolRegistry:
    """Maps AI function calls to concrete OBS and integration actions."""

    def __init__(
        self,
        obs: OBSController,
        assets_dir: Path,
        cough_recovery_seconds: float,
        source_lower_third_text: str,
        source_host_prompt_text: str,
        source_chat_question_text: str,
        source_sfx_airhorn: str,
        source_broll_image: str,
        allowed_scene_names: list[str] | tuple[str, ...] | None = None,
        scene_min_dwell_seconds: float = 0.0,
    ) -> None:
        self.obs = obs
        self.assets_dir = assets_dir
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.cough_recovery_seconds = cough_recovery_seconds
        self.source_lower_third_text = source_lower_third_text
        self.source_host_prompt_text = source_host_prompt_text
        self.source_chat_question_text = source_chat_question_text
        self.source_sfx_airhorn = source_sfx_airhorn
        self.source_broll_image = source_broll_image
        self.allowed_scene_names = set(allowed_scene_names or [])
        self.scene_min_dwell_seconds = max(0.0, float(scene_min_dwell_seconds))
        self._last_scene_switch_ts: float | None = None
        self._last_scene_name: str | None = None
        self._tools: dict[str, ToolFn] = {
            "switch_scene": self.switch_scene,
            "momentary_mute": self.momentary_mute,
            "mute_input": self.mute_input,
            "unmute_input": self.unmute_input,
            "show_lower_third": self.show_lower_third,
            "clear_lower_third": self.clear_lower_third,
            "show_host_prompt": self.show_host_prompt,
            "clear_host_prompt": self.clear_host_prompt,
            "toggle_overlay": self.toggle_overlay,
            "show_source_current_scene": self.show_source_current_scene,
            "hide_source_current_scene": self.hide_source_current_scene,
            "play_sfx": self.play_sfx,
            "inject_broll_from_url": self.inject_broll_from_url,
            "highlight_question": self.highlight_question,
            "clear_chat_question": self.clear_chat_question,
        }

    @property
    def tool_schemas(self) -> list[dict]:
        return [
            {"name": "switch_scene", "description": "Switch OBS program scene", "parameters": {"type": "object", "properties": {"scene_name": {"type": "string"}}, "required": ["scene_name"]}},
            {"name": "momentary_mute", "description": "Mute one OBS input temporarily", "parameters": {"type": "object", "properties": {"input_name": {"type": "string"}, "seconds": {"type": "number"}}, "required": ["input_name"]}},
            {"name": "mute_input", "description": "Mute an OBS audio input", "parameters": {"type": "object", "properties": {"input_name": {"type": "string"}}, "required": ["input_name"]}},
            {"name": "unmute_input", "description": "Unmute an OBS audio input", "parameters": {"type": "object", "properties": {"input_name": {"type": "string"}}, "required": ["input_name"]}},
            {"name": "show_lower_third", "description": "Set lower-third text source", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "source_name": {"type": "string"}}, "required": ["text"]}},
            {"name": "clear_lower_third", "description": "Clear lower-third text source", "parameters": {"type": "object", "properties": {"source_name": {"type": "string"}}}},
            {"name": "show_host_prompt", "description": "Set host prompt text source", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "source_name": {"type": "string"}}, "required": ["text"]}},
            {"name": "clear_host_prompt", "description": "Clear host prompt text source", "parameters": {"type": "object", "properties": {"source_name": {"type": "string"}}}},
            {"name": "toggle_overlay", "description": "Show or hide overlay source in a scene", "parameters": {"type": "object", "properties": {"scene_name": {"type": "string"}, "source_name": {"type": "string"}, "visible": {"type": "boolean"}}, "required": ["scene_name", "source_name", "visible"]}},
            {"name": "show_source_current_scene", "description": "Show a source in the current program scene", "parameters": {"type": "object", "properties": {"source_name": {"type": "string"}}, "required": ["source_name"]}},
            {"name": "hide_source_current_scene", "description": "Hide a source in the current program scene", "parameters": {"type": "object", "properties": {"source_name": {"type": "string"}}, "required": ["source_name"]}},
            {"name": "play_sfx", "description": "Play SFX media source", "parameters": {"type": "object", "properties": {"source_name": {"type": "string"}}, "required": ["source_name"]}},
            {"name": "inject_broll_from_url", "description": "Download image and route to OBS image source", "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "source_name": {"type": "string"}}, "required": ["url"]}},
            {"name": "highlight_question", "description": "Push highlighted chat question to text source", "parameters": {"type": "object", "properties": {"question": {"type": "string"}, "source_name": {"type": "string"}}, "required": ["question"]}},
            {"name": "clear_chat_question", "description": "Clear highlighted chat question text source", "parameters": {"type": "object", "properties": {"source_name": {"type": "string"}}}},
        ]

    async def execute(self, name: str, arguments: dict) -> dict:
        fn = self._tools.get(name)
        if fn is None:
            raise ValueError(f"Unknown tool: {name}")
        return await fn(arguments)

    async def switch_scene(self, args: dict) -> dict:
        scene_name = args.get("scene_name")
        if not isinstance(scene_name, str) or not scene_name:
            return {"ok": False, "error": "missing_or_invalid_scene_name"}
        if self.allowed_scene_names and scene_name not in self.allowed_scene_names:
            logger.warning("Rejected switch_scene for unknown scene: %s", scene_name)
            return {
                "ok": False,
                "error": "unknown_scene_name",
                "scene": scene_name,
                "allowed_scenes": sorted(self.allowed_scene_names),
            }

        now = asyncio.get_running_loop().time()
        if self._last_scene_name == scene_name:
            return {"ok": True, "scene": scene_name, "skipped": "already_active"}
        if self._last_scene_switch_ts is not None and self.scene_min_dwell_seconds > 0:
            age = now - self._last_scene_switch_ts
            if age < self.scene_min_dwell_seconds:
                return {
                    "ok": False,
                    "error": "scene_dwell_guard",
                    "retry_in_seconds": max(0.0, self.scene_min_dwell_seconds - age),
                }

        await self.obs.switch_scene(scene_name)
        self._last_scene_switch_ts = now
        self._last_scene_name = scene_name
        return {"ok": True, "scene": scene_name}

    async def momentary_mute(self, args: dict) -> dict:
        input_name = args["input_name"]
        seconds = float(args.get("seconds", self.cough_recovery_seconds))
        task = asyncio.create_task(self.obs.momentary_mute(input_name, seconds), name=f"momentary_mute:{input_name}")
        task.add_done_callback(self._log_task_error)
        return {"ok": True, "input": input_name, "seconds": seconds}

    async def mute_input(self, args: dict) -> dict:
        input_name = args["input_name"]
        await self.obs.set_mic_mute(input_name, True)
        return {"ok": True, "input": input_name, "muted": True}

    async def unmute_input(self, args: dict) -> dict:
        input_name = args["input_name"]
        await self.obs.set_mic_mute(input_name, False)
        return {"ok": True, "input": input_name, "muted": False}

    async def show_lower_third(self, args: dict) -> dict:
        text = args["text"]
        source_name = args.get("source_name", self.source_lower_third_text)
        if not isinstance(source_name, str) or source_name != self.source_lower_third_text:
            source_name = self.source_lower_third_text
        await self.obs.set_text_source(source_name, text)
        await self._ensure_source_visible_in_current_scene(source_name)
        return {"ok": True, "source": source_name}

    async def clear_lower_third(self, args: dict) -> dict:
        source_name = args.get("source_name", self.source_lower_third_text)
        if not isinstance(source_name, str) or source_name != self.source_lower_third_text:
            source_name = self.source_lower_third_text
        await self.obs.set_text_source(source_name, "")
        return {"ok": True, "source": source_name}

    async def show_host_prompt(self, args: dict) -> dict:
        text = args["text"]
        source_name = args.get("source_name", self.source_host_prompt_text)
        if not isinstance(source_name, str) or source_name != self.source_host_prompt_text:
            source_name = self.source_host_prompt_text
        await self.obs.set_text_source(source_name, text)
        await self._ensure_source_visible_in_current_scene(source_name)
        return {"ok": True, "source": source_name}

    async def clear_host_prompt(self, args: dict) -> dict:
        source_name = args.get("source_name", self.source_host_prompt_text)
        if not isinstance(source_name, str) or source_name != self.source_host_prompt_text:
            source_name = self.source_host_prompt_text
        await self.obs.set_text_source(source_name, "")
        return {"ok": True, "source": source_name}

    async def toggle_overlay(self, args: dict) -> dict:
        scene_name = args["scene_name"]
        source_name = args["source_name"]
        visible = bool(args["visible"])
        await self.obs.set_source_visible(scene_name, source_name, visible)
        return {"ok": True}

    async def show_source_current_scene(self, args: dict) -> dict:
        source_name = args["source_name"]
        scene_name = await self.obs.get_current_program_scene_name()
        await self.obs.set_source_visible(scene_name, source_name, True)
        return {"ok": True, "scene": scene_name, "source": source_name, "visible": True}

    async def hide_source_current_scene(self, args: dict) -> dict:
        source_name = args["source_name"]
        scene_name = await self.obs.get_current_program_scene_name()
        await self.obs.set_source_visible(scene_name, source_name, False)
        return {"ok": True, "scene": scene_name, "source": source_name, "visible": False}

    async def play_sfx(self, args: dict) -> dict:
        source_name = args.get("source_name", self.source_sfx_airhorn)
        if not isinstance(source_name, str) or source_name != self.source_sfx_airhorn:
            source_name = self.source_sfx_airhorn
        await self.obs.play_media_source(source_name)
        return {"ok": True, "source": source_name}

    async def inject_broll_from_url(self, args: dict) -> dict:
        url = args["url"]
        source_name = args.get("source_name", self.source_broll_image)
        if not isinstance(source_name, str) or source_name != self.source_broll_image:
            source_name = self.source_broll_image

        filename = url.split("?")[0].split("/")[-1] or "broll.jpg"
        local_path = self.assets_dir / filename

        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            local_path.write_bytes(response.content)

        await self.obs.set_image_source_file(source_name, local_path)
        return {"ok": True, "source": source_name, "path": str(local_path)}

    async def highlight_question(self, args: dict) -> dict:
        question = args["question"]
        source_name = args.get("source_name", self.source_chat_question_text)
        if not isinstance(source_name, str) or source_name != self.source_chat_question_text:
            source_name = self.source_chat_question_text
        await self.obs.set_text_source(source_name, question)
        await self._ensure_source_visible_in_current_scene(source_name)
        return {"ok": True, "source": source_name}

    async def clear_chat_question(self, args: dict) -> dict:
        source_name = args.get("source_name", self.source_chat_question_text)
        if not isinstance(source_name, str) or source_name != self.source_chat_question_text:
            source_name = self.source_chat_question_text
        await self.obs.set_text_source(source_name, "")
        return {"ok": True, "source": source_name}

    async def _ensure_source_visible_in_current_scene(self, source_name: str) -> None:
        try:
            scene_name = await self.obs.get_current_program_scene_name()
            await self.obs.set_source_visible(scene_name, source_name, True)
        except Exception:  # noqa: BLE001
            # Some sources may not exist in the current scene; ignore visibility failures.
            logger.debug("Could not force source %s visible in current scene", source_name, exc_info=True)

    @staticmethod
    def _log_task_error(task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception:  # noqa: BLE001
            logger.exception("Background task failed: %s", task.get_name())
