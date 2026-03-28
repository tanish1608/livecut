from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from obsws_python import ReqClient

logger = logging.getLogger(__name__)


class OBSController:
    """Thin async wrapper around obsws-python for low-latency control."""

    def __init__(self, host: str, port: int, password: str) -> None:
        self._host = host
        self._port = port
        self._password = password
        self._client: ReqClient | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        def _connect() -> ReqClient:
            return ReqClient(host=self._host, port=self._port, password=self._password, timeout=2)

        self._client = await asyncio.to_thread(_connect)
        logger.info("Connected to OBS websocket at %s:%s", self._host, self._port)

    async def list_scene_names(self) -> list[str]:
        async with self._lock:
            client = self._require_client()
            response = await asyncio.to_thread(client.get_scene_list)

        scenes_raw = getattr(response, "scenes", [])
        names: list[str] = []
        for scene in scenes_raw:
            if isinstance(scene, dict):
                scene_name = scene.get("sceneName")
            else:
                scene_name = getattr(scene, "sceneName", None)
            if isinstance(scene_name, str):
                names.append(scene_name)
        return names

    async def list_input_names(self) -> list[str]:
        async with self._lock:
            client = self._require_client()
            response = await asyncio.to_thread(client.get_input_list)

        inputs_raw = getattr(response, "inputs", [])
        names: list[str] = []
        for item in inputs_raw:
            if isinstance(item, dict):
                input_name = item.get("inputName")
            else:
                input_name = getattr(item, "inputName", None)
            if isinstance(input_name, str):
                names.append(input_name)
        return names

    async def validate_required_objects(
        self,
        required_scenes: list[str],
        required_inputs: list[str],
    ) -> dict[str, list[str] | Any]:
        scene_names = await self.list_scene_names()
        input_names = await self.list_input_names()

        scene_set = set(scene_names)
        input_set = set(input_names)

        missing_scenes = [name for name in required_scenes if name not in scene_set]
        missing_inputs = [name for name in required_inputs if name not in input_set]

        return {
            "missing_scenes": missing_scenes,
            "missing_inputs": missing_inputs,
            "available_scenes": scene_names,
            "available_inputs": input_names,
        }

    async def disconnect(self) -> None:
        if not self._client:
            return

        client = self._client
        self._client = None

        await asyncio.to_thread(client.base_client.ws.close)
        logger.info("Disconnected OBS websocket")

    async def switch_scene(self, scene_name: str) -> None:
        async with self._lock:
            client = self._require_client()
            await asyncio.to_thread(client.set_current_program_scene, scene_name)
            logger.info("Switched scene to %s", scene_name)

    async def set_mic_mute(self, input_name: str, muted: bool) -> None:
        async with self._lock:
            client = self._require_client()
            await asyncio.to_thread(client.set_input_mute, input_name, muted)
            logger.info("Set input %s muted=%s", input_name, muted)

    async def momentary_mute(self, input_name: str, seconds: float) -> None:
        await self.set_mic_mute(input_name, True)
        await asyncio.sleep(seconds)
        await self.set_mic_mute(input_name, False)

    async def set_source_visible(self, scene_name: str, source_name: str, visible: bool) -> None:
        async with self._lock:
            client = self._require_client()
            scene_items = await asyncio.to_thread(client.get_scene_item_list, scene_name)
            item_id = None
            for item in scene_items.scene_items:
                if item["sourceName"] == source_name:
                    item_id = item["sceneItemId"]
                    break
            if item_id is None:
                raise ValueError(f"Source {source_name} not found in scene {scene_name}")
            await asyncio.to_thread(client.set_scene_item_enabled, scene_name, item_id, visible)
            logger.info("Set source %s visible=%s in scene %s", source_name, visible, scene_name)

    async def set_image_source_file(self, source_name: str, image_path: Path) -> None:
        async with self._lock:
            client = self._require_client()
            settings = {"file": str(image_path)}
            await asyncio.to_thread(client.set_input_settings, source_name, settings, True)
            logger.info("Updated image source %s -> %s", source_name, image_path)

    async def set_text_source(self, source_name: str, text: str) -> None:
        async with self._lock:
            client = self._require_client()
            settings = {"text": text}
            await asyncio.to_thread(client.set_input_settings, source_name, settings, True)
            logger.info("Updated text source %s", source_name)

    async def play_media_source(self, source_name: str) -> None:
        async with self._lock:
            client = self._require_client()
            # Trigger a media source playback from start.
            await asyncio.to_thread(client.trigger_media_input_action, source_name, "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART")
            logger.info("Played media source %s", source_name)

    def _require_client(self) -> ReqClient:
        if not self._client:
            raise RuntimeError("OBS client is not connected")
        return self._client
