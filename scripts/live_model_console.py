from __future__ import annotations

import asyncio
import contextlib
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
import sounddevice as sd


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_modalities(value: str | None, model: str) -> list[str]:
    if value:
        modes = [part.strip().upper() for part in value.split(",") if part.strip()]
        if modes:
            return modes
    if "native-audio" in model.lower():
        return ["AUDIO"]
    return ["TEXT"]


def _build_client_from_env() -> tuple[genai.Client, dict[str, Any], str]:
    use_vertex = _as_bool(os.getenv("GOOGLE_GENAI_USE_VERTEXAI"), default=False)
    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    project = (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
    location = (os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1").strip()
    model = (os.getenv("LIVE_MODEL") or "gemini-3.1-flash-live-preview").strip()

    if use_vertex:
        if not project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for Vertex mode")
        client = genai.Client(vertexai=True, project=project, location=location)
        mode = f"vertex(project={project}, location={location})"
    else:
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for API-key mode")
        client = genai.Client(vertexai=False, api_key=api_key)
        mode = "api_key"

    modalities = _parse_modalities(os.getenv("LIVE_RESPONSE_MODALITIES"), model)
    cfg: dict[str, Any] = {
        "model": model,
        "modalities": modalities,
    }
    return client, cfg, mode


async def _receiver(session: Any) -> None:
    try:
        async for msg in session.receive():
            if getattr(msg, "setup_complete", False):
                print("[live] setup_complete")

            tool_call = getattr(msg, "tool_call", None)
            if tool_call is not None:
                fn_calls = list(getattr(tool_call, "function_calls", []) or [])
                for call in fn_calls:
                    print(f"[live] tool_call name={getattr(call, 'name', None)} args={getattr(call, 'args', None)}")

            server_content = getattr(msg, "server_content", None)
            if server_content is None:
                continue

            # Works for audio models when transcription is enabled.
            input_tx = getattr(server_content, "input_transcription", None)
            if input_tx is not None:
                text = getattr(input_tx, "text", None)
                if text:
                    print(f"[input_tx] {text}")

            output_tx = getattr(server_content, "output_transcription", None)
            if output_tx is not None:
                text = getattr(output_tx, "text", None)
                if text:
                    print(f"[assistant_tx] {text}")

            model_turn = getattr(server_content, "model_turn", None)
            if model_turn is not None:
                parts = list(getattr(model_turn, "parts", []) or [])
                for part in parts:
                    text = getattr(part, "text", None)
                    if text:
                        print(f"[assistant] {text}")
    except Exception as exc:  # noqa: BLE001
        print(f"[live] receiver error: {type(exc).__name__}: {exc}")
        raise
    finally:
        print("[live] receiver ended")


def _live_connect_config(modalities: list[str]) -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=modalities,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )


async def _open_session(client: genai.Client, model: str, modalities: list[str]) -> tuple[Any, asyncio.Task]:
    connect_config = _live_connect_config(modalities)
    session_cm = client.aio.live.connect(model=model, config=connect_config)
    session = await session_cm.__aenter__()
    receiver_task = asyncio.create_task(_receiver(session), name="live-receiver")
    print("[live] connected")
    return (session_cm, session, receiver_task)


def _list_input_devices() -> None:
    devices = sd.query_devices()
    default_input, _default_output = sd.default.device
    print("[audio] available input devices:")
    for idx, dev in enumerate(devices):
        max_in = int(dev.get("max_input_channels", 0))
        if max_in <= 0:
            continue
        mark = "*" if idx == default_input else " "
        name = str(dev.get("name", f"device_{idx}"))
        sr = dev.get("default_samplerate", "?")
        print(f"  {mark} {idx}: {name} (max_in={max_in}, default_sr={sr})")


async def _mic_sender_loop(session: Any, audio_queue: asyncio.Queue[bytes], sample_rate_hz: int) -> None:
    while True:
        chunk = await audio_queue.get()
        await session.send_realtime_input(
            audio=types.Blob(
                data=chunk,
                mime_type=f"audio/pcm;rate={sample_rate_hz}",
            )
        )


def _start_mic_stream(
    loop: asyncio.AbstractEventLoop,
    audio_queue: asyncio.Queue[bytes],
    sample_rate_hz: int,
    blocksize_frames: int,
    input_device: int | None,
) -> sd.RawInputStream:
    def callback(indata: bytes, frames: int, _time_info: Any, status: Any) -> None:
        if status:
            print(f"[audio] callback status: {status}")
        if frames <= 0:
            return

        def enqueue() -> None:
            if audio_queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    audio_queue.get_nowait()
            audio_queue.put_nowait(bytes(indata))

        loop.call_soon_threadsafe(enqueue)

    stream = sd.RawInputStream(
        samplerate=sample_rate_hz,
        blocksize=blocksize_frames,
        dtype="int16",
        channels=1,
        callback=callback,
        device=input_device,
    )
    stream.start()
    return stream


async def main() -> None:
    load_dotenv()

    client, cfg, mode = _build_client_from_env()
    model = cfg["model"]
    modalities = cfg["modalities"]

    sample_rate_hz = int(os.getenv("LIVE_MIC_SAMPLE_RATE_HZ", "16000"))
    blocksize_frames = int(os.getenv("LIVE_MIC_BLOCKSIZE", "1024"))
    input_device_env = (os.getenv("LIVE_MIC_DEVICE") or "").strip()
    input_device: int | None = int(input_device_env) if input_device_env else None

    print(f"Connecting live model={model} mode={mode} modalities={modalities}")
    print("Type messages and press Enter. Use /quit to exit.")
    print("Commands: /devices, /mic on, /mic off")

    session_cm: Any | None = None
    session: Any | None = None
    receiver_task: asyncio.Task | None = None
    mic_sender_task: asyncio.Task | None = None
    mic_stream: sd.RawInputStream | None = None
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)
    reconnect_lock = asyncio.Lock()
    shutting_down = False
    watchdog_task: asyncio.Task | None = None

    async def reconnect() -> None:
        nonlocal session_cm, session, receiver_task, mic_sender_task
        async with reconnect_lock:
            if shutting_down:
                return

            if session_cm is not None and receiver_task is not None and not receiver_task.done():
                return

            if receiver_task is not None:
                receiver_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receiver_task
                receiver_task = None
            if mic_sender_task is not None:
                mic_sender_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await mic_sender_task
                mic_sender_task = None
            if session_cm is not None:
                with contextlib.suppress(Exception):
                    await session_cm.__aexit__(None, None, None)
                session_cm = None
                session = None

            print("[live] reconnecting...")
            session_cm, session, receiver_task = await _open_session(client, model, modalities)
            if mic_stream is not None:
                mic_sender_task = asyncio.create_task(_mic_sender_loop(session, audio_queue, sample_rate_hz), name="mic-sender")

    async def reconnect_watchdog() -> None:
        while not shutting_down:
            await asyncio.sleep(1.0)
            if receiver_task is not None and receiver_task.done():
                print("[live] stream ended, watchdog reconnect")
                with contextlib.suppress(Exception):
                    await reconnect()

    try:
        session_cm, session, receiver_task = await _open_session(client, model, modalities)
        watchdog_task = asyncio.create_task(reconnect_watchdog(), name="reconnect-watchdog")
        while True:
            user_text = await asyncio.to_thread(input, "you> ")
            if user_text.strip().lower() in {"/quit", "/exit"}:
                print("Exiting...")
                break
            if user_text.strip().lower() == "/devices":
                _list_input_devices()
                continue
            if user_text.strip().lower() == "/mic on":
                if mic_stream is None:
                    if receiver_task is not None and receiver_task.done():
                        await reconnect()
                    loop = asyncio.get_running_loop()
                    mic_stream = _start_mic_stream(
                        loop=loop,
                        audio_queue=audio_queue,
                        sample_rate_hz=sample_rate_hz,
                        blocksize_frames=blocksize_frames,
                        input_device=input_device,
                    )
                    mic_sender_task = asyncio.create_task(_mic_sender_loop(session, audio_queue, sample_rate_hz), name="mic-sender")
                    print("[audio] mic streaming ON")
                else:
                    print("[audio] mic already ON")
                continue
            if user_text.strip().lower() == "/mic off":
                if mic_sender_task is not None:
                    mic_sender_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await mic_sender_task
                    mic_sender_task = None
                if mic_stream is not None:
                    with contextlib.suppress(Exception):
                        mic_stream.stop()
                        mic_stream.close()
                    mic_stream = None
                print("[audio] mic streaming OFF")
                continue
            if not user_text.strip():
                continue

            if receiver_task is not None and receiver_task.done():
                print("[live] session closed after previous turn")
                await reconnect()

            try:
                await session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text=user_text)]),
                    turn_complete=True,
                )
                print("[live] sent")
            except Exception as exc:  # noqa: BLE001
                print(f"[live] send failed ({type(exc).__name__}): {exc}")
                await reconnect()
                await session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text=user_text)]),
                    turn_complete=True,
                )
                print("[live] sent after reconnect")
    finally:
        shutting_down = True
        if watchdog_task is not None:
            watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog_task
        if mic_sender_task is not None:
            mic_sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await mic_sender_task
        if mic_stream is not None:
            with contextlib.suppress(Exception):
                mic_stream.stop()
                mic_stream.close()
        if receiver_task is not None:
            receiver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver_task
        if session_cm is not None:
            with contextlib.suppress(Exception):
                await session_cm.__aexit__(None, None, None)

    client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
