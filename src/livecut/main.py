from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable

from dotenv import load_dotenv
from google import genai
from google.auth.exceptions import DefaultCredentialsError

from .agent_runtime import LiveCutRuntime
from .config import settings
from .gemini_bridge import GeminiLiveBridge
from .obs_controller import OBSController
from .simple_assistant import AssistantConfig, ChromeWakeListener, SimpleVoiceAssistant
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


def build_response_modalities() -> list[str] | None:
    explicit = parse_response_modalities(settings.live_response_modalities)
    if explicit is not None:
        return explicit
    if settings.gemini_voice_assistant_mode:
        # Native-audio live models require AUDIO response modality.
        if "native-audio" in settings.live_model.lower():
            return ["AUDIO"]
        return ["TEXT"]
    return None


def parse_csv_items(value: str | None) -> list[str] | None:
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    items = [part for part in parts if part]
    return items or None


def build_live_system_instruction() -> str:
    base = f"""
You are LiveCut execution agent. You receive context from the NVIDIA live director and execute OBS tools.

Primary goals:
- Execute actions only when necessary and high confidence.
- Prioritize user voice commands and explicit operator intent.
- Use NVIDIA director context as advisory input for autonomous actions.

Tooling rules:
- Prefer tool calls over free-form text when an action is needed.
- Use exact OBS names provided in context messages.
- Do not invent source names or scene names.
- Keep actions sparse and deliberate: avoid rapid scene thrashing.
- If uncertain, ask for a clarification in text rather than issuing risky tool calls.
- When user asks for visuals/images, use inject_broll_from_url.

Scenario policy:
- NVIDIA live director owns continuous visual narration and kill detection.
- You own tool execution and command fulfillment.
- For chat scene, prioritize highlight_question and lower-third/comment actions.
- For kill/cough safety, use play_sfx and momentary_mute only when warranted.

Reliability:
- Sessions may reconnect; on reconnect, continue operating from latest context and avoid duplicate spam actions.
""".strip()

    if settings.gemini_voice_assistant_mode:
        wake_word = settings.gemini_voice_wake_word.strip() or "gemini"
        if settings.gemini_require_wake_word:
            base = (
                f"{base}\n\n"
                "Voice assistant command mode:\n"
                f"- Only execute tool calls when the host command includes wake word '{wake_word}'.\n"
                f"- Tool calls are allowed for about {settings.gemini_wake_window_seconds:.0f} seconds after wake word detection.\n"
                "- If wake word is missing, do not call tools.\n"
                "- Prefer direct command fulfillment: switch scenes, show overlays, highlight chat, inject images.\n"
                "- Confirm actions in short spoken replies.\n"
                "- Keep confirmations short and action-focused."
            )
        else:
            base = (
                f"{base}\n\n"
                "Voice assistant command mode:\n"
                "- Treat host spoken imperative commands as highest priority.\n"
                "- Prefer direct command fulfillment with tool calls.\n"
                "- Keep confirmations short and action-focused."
            )

    if settings.live_system_instruction:
        return f"{base}\n\nUser override instruction:\n{settings.live_system_instruction.strip()}"
    return base


def build_vlm_system_instruction() -> str:
    base = """
You are the NVIDIA live director for a livestream control room.

You do NOT directly control OBS unless explicitly configured.
Your primary role is:
- describe what is happening on stream succinctly,
- detect probable kill/elimination moments,
- indicate whether Gemini should take an action,
- suggest high-level requested actions (not strict tool calls).

Return stable, conservative outputs and avoid noisy flip-flopping.
""".strip()

    if settings.vlm_system_instruction:
        return f"{base}\n\nUser override instruction:\n{settings.vlm_system_instruction.strip()}"
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
        settings.source_broll_image,
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
        source_broll_image=settings.source_broll_image,
        allowed_scene_names=[settings.scene_gameplay_focus, settings.scene_chatting_focus],
        scene_min_dwell_seconds=settings.scene_min_dwell_seconds,
    )

    if settings.gemini_simple_assistant_mode:
        client_kwargs: dict[str, object] = {"vertexai": settings.google_genai_use_vertexai}
        if settings.google_genai_use_vertexai:
            client_kwargs["project"] = settings.google_cloud_project
            client_kwargs["location"] = settings.google_cloud_location
        else:
            client_kwargs["api_key"] = settings.google_api_key

        try:
            client = genai.Client(**client_kwargs)
        except DefaultCredentialsError:
            if not settings.google_api_key:
                await obs.disconnect()
                raise
            client = genai.Client(vertexai=False, api_key=settings.google_api_key)

        listener = ChromeWakeListener(
            host=settings.chrome_listener_host,
            port=settings.chrome_listener_port,
            auto_open=settings.chrome_auto_open,
        )
        assistant = SimpleVoiceAssistant(
            tools=tools,
            listener=listener,
            config=AssistantConfig(
                wake_word=settings.gemini_voice_wake_word,
                command_model=settings.gemini_command_model,
                speak_replies=settings.gemini_speak_replies,
            ),
            client=client,
            scene_gameplay_focus=settings.scene_gameplay_focus,
            scene_chatting_focus=settings.scene_chatting_focus,
            source_sfx_airhorn=settings.source_sfx_airhorn,
            source_host_prompt_text=settings.source_host_prompt_text,
            source_chat_question_text=settings.source_chat_question_text,
            source_broll_image=settings.source_broll_image,
        )

        try:
            await assistant.run()
        finally:
            client.close()
            await obs.disconnect()
        return

    gemini_bridge = (
        GeminiLiveBridge(
            model=settings.live_model,
            tool_schemas=tools.tool_schemas,
            use_vertexai=settings.google_genai_use_vertexai,
            api_key=settings.google_api_key,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
            system_instruction=build_live_system_instruction(),
            response_modalities=build_response_modalities(),
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
            audio_output_enabled=settings.gemini_audio_output_enabled,
            audio_output_device=settings.gemini_audio_output_device,
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
            role=settings.vlm_role,
            enable_tool_calls=settings.vlm_enable_tool_calls,
            kill_detection_enabled=settings.vlm_kill_detection_enabled,
            kill_keywords=parse_csv_items(settings.vlm_kill_keywords),
            system_instruction=build_vlm_system_instruction(),
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
        gemini_use_vlm_context=settings.gemini_use_vlm_context,
        gemini_vlm_context_min_interval_seconds=settings.gemini_vlm_context_min_interval_seconds,
        gemini_chat_actions_only_in_chat_scene=settings.gemini_chat_actions_only_in_chat_scene,
        gemini_require_wake_word=settings.gemini_require_wake_word,
        gemini_voice_wake_word=settings.gemini_voice_wake_word,
        gemini_wake_window_seconds=settings.gemini_wake_window_seconds,
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
