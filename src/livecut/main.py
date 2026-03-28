from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable

from dotenv import load_dotenv

from .agent_runtime import LiveCutRuntime
from .config import settings
from .gemini_bridge import GeminiLiveBridge
from .obs_controller import OBSController
from .tools import ToolRegistry
from .vlm_bridge import NvidiaVLMBridge


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_response_modalities(value: str | None) -> list[str] | None:
    if not value:
        return None
    parts = [part.strip().upper() for part in value.split(",")]
    modes = [part for part in parts if part]
    return modes or None


def parse_csv_items(value: str | None) -> list[str] | None:
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    items = [part for part in parts if part]
    return items or None


def build_live_system_instruction() -> str:
    base = f"""
You are LiveCut, an autonomous real-time stream producer controlling OBS through function calls.

Primary goals:
- Maintain smooth pacing and viewer clarity.
- Keep audio quality safe (react to cough/spike events and speech context).
- Prioritize scene switching and overlays based on live multimodal context.

Tooling rules:
- Prefer tool calls over free-form text when an action is needed.
- Use exact OBS names provided in context messages.
- Do not invent source names or scene names.
- Keep actions sparse and deliberate: avoid rapid scene thrashing.
- If uncertain, ask for a clarification in text rather than issuing risky tool calls.

Broadcast policy:
- When gameplay action rises, favor gameplay focus.
- During menu/chat/static moments, favor chatting focus.
- If host username appears in killfeed, trigger SFX.
- For high-quality audience questions, use highlight_question.

Reliability:
- Sessions may reconnect; on reconnect, continue operating from latest context and avoid duplicate spam actions.
""".strip()

    if settings.live_system_instruction:
        return f"{base}\n\nUser override instruction:\n{settings.live_system_instruction.strip()}"
    return base


def build_video_frame_provider(obs: OBSController) -> Callable[[], Awaitable[bytes | None]] | None:
    mode = settings.gemini_video_source_mode.strip().lower()
    if mode == "camera_device":
        return None

    async def _provider() -> bytes | None:
        source_name = settings.gemini_video_source_name
        if not source_name:
            source_name = await obs.get_current_program_scene_name()

        try:
            return await obs.get_source_screenshot_jpeg(
                source_name=source_name,
                width=settings.gemini_video_width,
                height=settings.gemini_video_height,
                quality=settings.gemini_video_jpeg_quality,
            )
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).exception("Failed to capture OBS frame for source=%s", source_name)
            return None

    return _provider


def build_obs_frame_provider(obs: OBSController) -> Callable[[], Awaitable[bytes | None]]:
    async def _provider() -> bytes | None:
        source_name = settings.gemini_video_source_name
        if not source_name:
            source_name = await obs.get_current_program_scene_name()

        try:
            return await obs.get_source_screenshot_jpeg(
                source_name=source_name,
                width=settings.gemini_video_width,
                height=settings.gemini_video_height,
                quality=settings.gemini_video_jpeg_quality,
            )
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).exception("Failed to capture OBS frame for VLM source=%s", source_name)
            return None

    return _provider


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
        allowed_scene_names=[settings.scene_gameplay_focus, settings.scene_chatting_focus],
        scene_min_dwell_seconds=settings.scene_min_dwell_seconds,
    )

    gemini_bridge = (
        GeminiLiveBridge(
            model=settings.live_model,
            tool_schemas=tools.tool_schemas,
            use_vertexai=settings.google_genai_use_vertexai,
            api_key=settings.google_api_key,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
            system_instruction=build_live_system_instruction(),
            response_modalities=parse_response_modalities(settings.live_response_modalities),
            receive_idle_log_seconds=settings.gemini_receive_idle_log_seconds,
            bootstrap_user_text=settings.gemini_bootstrap_user_text,
            keepalive_seconds=settings.gemini_keepalive_seconds,
            keepalive_text=settings.gemini_keepalive_text,
            auto_reconnect=settings.gemini_auto_reconnect,
            reconnect_backoff_seconds=settings.gemini_reconnect_backoff_seconds,
            max_reconnect_backoff_seconds=settings.gemini_max_reconnect_backoff_seconds,
            audio_enabled=settings.gemini_audio_enabled,
            video_enabled=settings.gemini_video_enabled,
            audio_sample_rate_hz=settings.gemini_audio_sample_rate_hz,
            audio_blocksize_frames=settings.gemini_audio_blocksize_frames,
            audio_input_device=settings.gemini_audio_input_device,
            video_device_index=settings.gemini_video_device_index,
            video_fps=settings.gemini_video_fps,
            video_width=settings.gemini_video_width,
            video_height=settings.gemini_video_height,
            video_jpeg_quality=settings.gemini_video_jpeg_quality,
            video_frame_provider=build_video_frame_provider(obs),
        )
        if settings.enable_gemini
        else None
    )

    vlm_bridge = (
        NvidiaVLMBridge(
            model=settings.vlm_model,
            base_url=settings.vlm_base_url,
            api_key=settings.nvidia_api_key,
            frame_provider=build_obs_frame_provider(obs),
            tool_schemas=tools.tool_schemas,
            scene_gameplay_focus=settings.scene_gameplay_focus,
            scene_chatting_focus=settings.scene_chatting_focus,
            source_sfx_airhorn=settings.source_sfx_airhorn,
            source_host_prompt_text=settings.source_host_prompt_text,
            source_chat_question_text=settings.source_chat_question_text,
            poll_seconds=settings.vlm_poll_seconds,
            request_timeout_seconds=settings.vlm_request_timeout_seconds,
            max_actions_per_turn=settings.vlm_max_actions_per_turn,
            allowed_tool_names=parse_csv_items(settings.vlm_allowed_tools),
            action_cooldown_seconds=settings.vlm_action_cooldown_seconds,
            error_backoff_base_seconds=settings.vlm_error_backoff_base_seconds,
            error_backoff_max_seconds=settings.vlm_error_backoff_max_seconds,
            system_instruction=settings.vlm_system_instruction,
        )
        if settings.enable_vlm
        else None
    )

    runtime = LiveCutRuntime(
        tools=tools,
        host_username=settings.host_username,
        segment_minutes=settings.segment_max_minutes,
        chat_seconds=settings.chat_batch_seconds,
        ai_only_scene_switching=settings.ai_only_scene_switching,
        gemini_scene_switch_delay_seconds=settings.gemini_scene_switch_delay_seconds,
        vlm_scene_switch_delay_seconds=settings.vlm_scene_switch_delay_seconds,
        scene_gameplay_focus=settings.scene_gameplay_focus,
        scene_chatting_focus=settings.scene_chatting_focus,
        input_host_mic=settings.input_host_mic,
        source_sfx_airhorn=settings.source_sfx_airhorn,
        source_host_prompt_text=settings.source_host_prompt_text,
        status_log_seconds=settings.runtime_status_log_seconds,
        run_simulation_loops_with_gemini=settings.run_simulation_loops_with_gemini,
        gemini_bridge=gemini_bridge,
        vlm_bridge=vlm_bridge,
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
