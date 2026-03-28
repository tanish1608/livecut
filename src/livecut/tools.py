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
        source_chat_question_text: str,
        source_sfx_airhorn: str,
    ) -> None:
        self.obs = obs
        self.assets_dir = assets_dir
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.cough_recovery_seconds = cough_recovery_seconds
        self.source_lower_third_text = source_lower_third_text
        self.source_chat_question_text = source_chat_question_text
        self.source_sfx_airhorn = source_sfx_airhorn
        self._tools: dict[str, ToolFn] = {
            "switch_scene": self.switch_scene,
            "momentary_mute": self.momentary_mute,
            "show_lower_third": self.show_lower_third,
            "toggle_overlay": self.toggle_overlay,
            "play_sfx": self.play_sfx,
            "inject_broll_from_url": self.inject_broll_from_url,
            "highlight_question": self.highlight_question,
        }

    @property
    def tool_schemas(self) -> list[dict]:
        return [
            {"name": "switch_scene", "description": "Switch OBS program scene", "parameters": {"type": "object", "properties": {"scene_name": {"type": "string"}}, "required": ["scene_name"]}},
            {"name": "momentary_mute", "description": "Mute one OBS input temporarily", "parameters": {"type": "object", "properties": {"input_name": {"type": "string"}, "seconds": {"type": "number"}}, "required": ["input_name"]}},
            {"name": "show_lower_third", "description": "Set lower-third text source", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "source_name": {"type": "string"}}, "required": ["text"]}},
            {"name": "toggle_overlay", "description": "Show or hide overlay source in a scene", "parameters": {"type": "object", "properties": {"scene_name": {"type": "string"}, "source_name": {"type": "string"}, "visible": {"type": "boolean"}}, "required": ["scene_name", "source_name", "visible"]}},
            {"name": "play_sfx", "description": "Play SFX media source", "parameters": {"type": "object", "properties": {"source_name": {"type": "string"}}, "required": ["source_name"]}},
            {"name": "inject_broll_from_url", "description": "Download image and route to OBS image source", "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "source_name": {"type": "string"}}, "required": ["url", "source_name"]}},
            {"name": "highlight_question", "description": "Push highlighted chat question to text source", "parameters": {"type": "object", "properties": {"question": {"type": "string"}, "source_name": {"type": "string"}}, "required": ["question"]}},
        ]

    async def execute(self, name: str, arguments: dict) -> dict:
        fn = self._tools.get(name)
        if fn is None:
            raise ValueError(f"Unknown tool: {name}")
        return await fn(arguments)

    async def switch_scene(self, args: dict) -> dict:
        scene_name = args["scene_name"]
        await self.obs.switch_scene(scene_name)
        return {"ok": True, "scene": scene_name}

    async def momentary_mute(self, args: dict) -> dict:
        input_name = args["input_name"]
        seconds = float(args.get("seconds", self.cough_recovery_seconds))
        task = asyncio.create_task(self.obs.momentary_mute(input_name, seconds), name=f"momentary_mute:{input_name}")
        task.add_done_callback(self._log_task_error)
        return {"ok": True, "input": input_name, "seconds": seconds}

    async def show_lower_third(self, args: dict) -> dict:
        text = args["text"]
        source_name = args.get("source_name", self.source_lower_third_text)
        await self.obs.set_text_source(source_name, text)
        return {"ok": True, "source": source_name}

    async def toggle_overlay(self, args: dict) -> dict:
        scene_name = args["scene_name"]
        source_name = args["source_name"]
        visible = bool(args["visible"])
        await self.obs.set_source_visible(scene_name, source_name, visible)
        return {"ok": True}

    async def play_sfx(self, args: dict) -> dict:
        source_name = args.get("source_name", self.source_sfx_airhorn)
        await self.obs.play_media_source(source_name)
        return {"ok": True, "source": source_name}

    async def inject_broll_from_url(self, args: dict) -> dict:
        url = args["url"]
        source_name = args["source_name"]

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
        await self.obs.set_text_source(source_name, question)
        return {"ok": True, "source": source_name}

    @staticmethod
    def _log_task_error(task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception:  # noqa: BLE001
            logger.exception("Background task failed: %s", task.get_name())
