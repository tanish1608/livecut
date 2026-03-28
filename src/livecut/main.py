from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

from .agent_runtime import LiveCutRuntime
from .config import settings
from .gemini_bridge import GeminiLiveBridge
from .obs_controller import OBSController
from .tools import ToolRegistry


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


async def run() -> None:
    load_dotenv()
    configure_logging()

    obs = OBSController(
        host=settings.obs_host,
        port=settings.obs_port,
        password=settings.obs_password,
    )
    await obs.connect()

    required_scenes = [
        settings.scene_gameplay_focus,
        settings.scene_chatting_focus,
    ]
    required_inputs = [
        settings.input_host_mic,
        settings.source_sfx_airhorn,
        settings.source_lower_third_text,
        settings.source_host_prompt_text,
        settings.source_chat_question_text,
    ]

    validation = await obs.validate_required_objects(required_scenes, required_inputs)
    missing_scenes = validation["missing_scenes"]
    missing_inputs = validation["missing_inputs"]
    if missing_scenes or missing_inputs:
        logging.getLogger(__name__).error("Missing OBS scenes: %s", missing_scenes)
        logging.getLogger(__name__).error("Missing OBS inputs: %s", missing_inputs)
        logging.getLogger(__name__).info("Available OBS scenes: %s", validation["available_scenes"])
        logging.getLogger(__name__).info("Available OBS inputs: %s", validation["available_inputs"])
        await obs.disconnect()
        raise RuntimeError("OBS validation failed. Fix scene/input names in .env or OBS and retry.")

    tools = ToolRegistry(
        obs=obs,
        assets_dir=Path("assets/broll"),
        cough_recovery_seconds=settings.cough_recovery_seconds,
        source_lower_third_text=settings.source_lower_third_text,
        source_chat_question_text=settings.source_chat_question_text,
        source_sfx_airhorn=settings.source_sfx_airhorn,
    )

    gemini_bridge = GeminiLiveBridge(settings.live_model) if settings.enable_gemini else None

    runtime = LiveCutRuntime(
        tools=tools,
        host_username=settings.host_username,
        segment_minutes=settings.segment_max_minutes,
        chat_seconds=settings.chat_batch_seconds,
        scene_gameplay_focus=settings.scene_gameplay_focus,
        scene_chatting_focus=settings.scene_chatting_focus,
        input_host_mic=settings.input_host_mic,
        source_sfx_airhorn=settings.source_sfx_airhorn,
        source_host_prompt_text=settings.source_host_prompt_text,
        gemini_bridge=gemini_bridge,
    )

    await runtime.start()

    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await runtime.stop()
        await obs.disconnect()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
