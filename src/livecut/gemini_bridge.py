from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from collections.abc import AsyncIterator
from collections.abc import Awaitable, Callable
from typing import Any

from google import genai
from google.genai import types
from google.auth.exceptions import DefaultCredentialsError

try:
    import sounddevice as sd
except Exception:  # noqa: BLE001
    sd = None

try:
    import cv2  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    cv2 = None

from .types import StreamSignal

logger = logging.getLogger(__name__)


class GeminiLiveBridge:
    """Adapter boundary for Gemini Multimodal Live API integration.

    This class is intentionally small so you can swap in the exact SDK flow
    you use (Vertex AI or Gemini API key auth) without changing the runtime.
    """

    def __init__(
        self,
        model: str,
        tool_schemas: list[dict[str, Any]],
        use_vertexai: bool,
        api_key: str | None,
        project: str | None,
        location: str,
        system_instruction: str | None = None,
        response_modalities: list[str] | None = None,
        receive_idle_log_seconds: int = 15,
        bootstrap_user_text: str | None = None,
        keepalive_seconds: int = 0,
        keepalive_text: str = "status",
        auto_reconnect: bool = True,
        reconnect_backoff_seconds: float = 2.0,
        max_reconnect_backoff_seconds: float = 30.0,
        audio_enabled: bool = True,
        video_enabled: bool = False,
        audio_sample_rate_hz: int = 16000,
        audio_blocksize_frames: int = 1024,
        audio_input_device: str | None = None,
        video_device_index: int = 0,
        video_fps: float = 2.0,
        video_width: int = 1280,
        video_height: int = 720,
        video_jpeg_quality: int = 70,
        video_frame_provider: Callable[[], Awaitable[bytes | None]] | None = None,
    ) -> None:
        self.model = model
        self.tool_schemas = tool_schemas
        self.use_vertexai = use_vertexai
        self.api_key = api_key
        self.project = project
        self.location = location
        self.system_instruction = system_instruction
        self.response_modalities = response_modalities
        self.receive_idle_log_seconds = max(5, int(receive_idle_log_seconds))
        self.bootstrap_user_text = bootstrap_user_text
        self.keepalive_seconds = max(0, int(keepalive_seconds))
        self.keepalive_text = keepalive_text
        self.auto_reconnect = auto_reconnect
        self.reconnect_backoff_seconds = max(0.5, float(reconnect_backoff_seconds))
        self.max_reconnect_backoff_seconds = max(self.reconnect_backoff_seconds, float(max_reconnect_backoff_seconds))
        self.audio_enabled = audio_enabled
        self.video_enabled = video_enabled
        self.audio_sample_rate_hz = max(8000, int(audio_sample_rate_hz))
        self.audio_blocksize_frames = max(256, int(audio_blocksize_frames))
        self.audio_input_device = self._parse_audio_device(audio_input_device)
        self.video_device_index = int(video_device_index)
        self.video_fps = max(0.2, float(video_fps))
        self.video_width = max(160, int(video_width))
        self.video_height = max(120, int(video_height))
        self.video_jpeg_quality = min(95, max(30, int(video_jpeg_quality)))
        self.video_frame_provider = video_frame_provider

        self._client: genai.Client | None = None
        self._live_cm: Any | None = None
        self._session: Any | None = None
        self._connected = False
        self._receive_message_count = 0
        self._audio_queue: asyncio.Queue[bytes] | None = None
        self._audio_stream: Any | None = None
        self._media_tasks: list[asyncio.Task] = []
        self._bootstrap_sent = False

    async def connect(self) -> None:
        if self.use_vertexai and not self.project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is required when GOOGLE_GENAI_USE_VERTEXAI=true")
        if not self.use_vertexai and not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is required when GOOGLE_GENAI_USE_VERTEXAI=false")

        client_kwargs: dict[str, Any] = {"vertexai": self.use_vertexai}
        if self.use_vertexai:
            client_kwargs["project"] = self.project
            client_kwargs["location"] = self.location
        else:
            client_kwargs["api_key"] = self.api_key

        try:
            self._client = genai.Client(**client_kwargs)
            if self.use_vertexai:
                logger.info("Gemini bridge using Vertex AI project=%s location=%s", self.project, self.location)
        except DefaultCredentialsError:
            if not self.use_vertexai or not self.api_key:
                raise
            logger.warning("Vertex AI ADC credentials unavailable. Falling back to API-key mode for Gemini live.")
            self._client = genai.Client(vertexai=False, api_key=self.api_key)

        config = types.LiveConnectConfig(
            response_modalities=self._effective_response_modalities(),
            tools=[self._build_tool_declaration()],
            system_instruction=self.system_instruction,
        )

        try:
            await self._open_session(config)
        except Exception as exc:  # noqa: BLE001
            mode = "vertex" if self.use_vertexai else "api_key"
            self._live_cm = None
            self._session = None
            raise RuntimeError(
                f"Gemini Live connect failed for model={self.model} mode={mode}. "
                "If mode=vertex, set up gcloud ADC (application-default login). "
                "If mode=api_key, verify Live API access for this project/key and model availability."
            ) from exc
        self._connected = True
        logger.info(
            "Gemini bridge initialized for model=%s modalities=%s",
            self.model,
            self._effective_response_modalities(),
        )
        logger.info(
            "Gemini realtime input | audio_enabled=%s video_enabled=%s video_mode=%s audio_rate=%s video_fps=%.2f",
            self.audio_enabled,
            self.video_enabled,
            "obs_feed" if self.video_frame_provider is not None else "camera_device",
            self.audio_sample_rate_hz,
            self.video_fps,
        )

    async def disconnect(self) -> None:
        await self._close_session()
        if self._client is not None:
            self._client.close()

        self._live_cm = None
        self._session = None
        self._client = None
        self._connected = False

    async def inject_system_message(self, text: str) -> None:
        session = self._require_session()
        await session.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=text)]), turn_complete=True)

    async def send_user_text(self, text: str) -> None:
        session = self._require_session()
        await session.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=text)]), turn_complete=True)

    async def send_tool_result(self, call_id: str, tool_name: str, result: dict[str, Any]) -> None:
        session = self._require_session()
        response = types.FunctionResponse(
            id=call_id,
            name=tool_name,
            response=result,
        )
        await session.send_tool_response(function_responses=[response])

    async def signals(self) -> AsyncIterator[StreamSignal]:
        """Yield structured stream events from Gemini live outputs.

        Expected outputs include speech activity, scene context switches, tool-call
        intents, and entity extraction events.
        """
        receive_iter = self._require_session().receive().__aiter__()
        reconnect_delay = self.reconnect_backoff_seconds

        while self._connected:
            try:
                msg = await asyncio.wait_for(receive_iter.__anext__(), timeout=float(self.receive_idle_log_seconds))
            except TimeoutError:
                logger.info(
                    "Gemini receive idle for %ss. Waiting for model/server events (tool calls, transcription, etc.).",
                    self.receive_idle_log_seconds,
                )
                if self.keepalive_seconds > 0:
                    await self._maybe_send_keepalive()
                continue
            except StopAsyncIteration:
                logger.warning("Gemini receive stream ended")
                if not self.auto_reconnect:
                    return
                reconnected = await self._attempt_reconnect(reconnect_delay)
                if not reconnected:
                    return
                reconnect_delay = min(reconnect_delay * 2.0, self.max_reconnect_backoff_seconds)
                receive_iter = self._require_session().receive().__aiter__()
                continue
            except Exception:  # noqa: BLE001
                logger.exception("Gemini receive loop failed unexpectedly")
                if not self.auto_reconnect:
                    return
                reconnected = await self._attempt_reconnect(reconnect_delay)
                if not reconnected:
                    return
                reconnect_delay = min(reconnect_delay * 2.0, self.max_reconnect_backoff_seconds)
                receive_iter = self._require_session().receive().__aiter__()
                continue

            reconnect_delay = self.reconnect_backoff_seconds

            self._receive_message_count += 1
            if self._receive_message_count % 25 == 0:
                logger.info("Gemini receive heartbeat | messages=%d", self._receive_message_count)

            setup_complete = bool(getattr(msg, "setup_complete", False))
            if setup_complete:
                logger.info("Gemini setup complete event received")
                yield StreamSignal(source="gemini", kind="setup_complete", payload={})

            tool_call = getattr(msg, "tool_call", None)
            if tool_call is not None:
                function_calls = list(getattr(tool_call, "function_calls", []) or [])
                logger.info("Gemini tool-call event with %d function call(s)", len(function_calls))
                for fn_call in function_calls:
                    call_name = getattr(fn_call, "name", None)
                    if not call_name:
                        continue
                    call_id = getattr(fn_call, "id", None)
                    args = getattr(fn_call, "args", None) or {}
                    if not isinstance(args, dict):
                        args = {}
                    yield StreamSignal(
                        source="gemini",
                        kind="tool_call",
                        payload={"id": call_id, "name": call_name, "arguments": args},
                    )

            server_content = getattr(msg, "server_content", None)
            if server_content is not None:
                input_tx = getattr(server_content, "input_transcription", None)
                if input_tx is not None:
                    text = getattr(input_tx, "text", None)
                    if text:
                        logger.info("Gemini input transcription received (%d chars)", len(text))
                        yield StreamSignal(source="audio", kind="speech", payload={"speaker": "host", "text": text})

                output_tx = getattr(server_content, "output_transcription", None)
                if output_tx is not None:
                    text = getattr(output_tx, "text", None)
                    if text:
                        logger.info("Gemini output transcription received (%d chars)", len(text))
                        yield StreamSignal(source="gemini", kind="assistant_transcript", payload={"text": text})

            voice_activity = getattr(msg, "voice_activity", None)
            if voice_activity is not None:
                activity_type = str(getattr(voice_activity, "voice_activity_type", "")).lower()
                if "start" in activity_type:
                    yield StreamSignal(source="audio", kind="speech_start", payload={"speaker": "host"})
                elif "end" in activity_type:
                    yield StreamSignal(source="audio", kind="speech_end", payload={"speaker": "host"})

    def _build_tool_declaration(self) -> types.Tool:
        function_declarations: list[types.FunctionDeclaration] = []
        for schema in self.tool_schemas:
            name = schema.get("name")
            if not isinstance(name, str) or not name:
                continue
            function_declarations.append(
                types.FunctionDeclaration(
                    name=name,
                    description=schema.get("description"),
                    parameters_json_schema=schema.get("parameters", {"type": "object"}),
                )
            )

        return types.Tool(function_declarations=function_declarations)

    def _effective_response_modalities(self) -> list[str]:
        if self.response_modalities:
            return self.response_modalities
        if "native-audio" in self.model.lower():
            return ["AUDIO"]
        return ["TEXT"]

    async def _open_session(self, config: types.LiveConnectConfig) -> None:
        self._live_cm = self._client.aio.live.connect(model=self.model, config=config)
        self._session = await self._live_cm.__aenter__()
        if self.bootstrap_user_text and not self._bootstrap_sent:
            await self._session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=self.bootstrap_user_text)]),
                turn_complete=True,
            )
            self._bootstrap_sent = True
            logger.info("Gemini bootstrap prompt sent")
        await self._start_realtime_inputs(self._session)

    async def _close_session(self) -> None:
        await self._stop_realtime_inputs()
        if self._live_cm is not None:
            with suppress(Exception):
                await self._live_cm.__aexit__(None, None, None)
        self._live_cm = None
        self._session = None

    async def _attempt_reconnect(self, delay_seconds: float) -> bool:
        if not self._connected:
            return False
        logger.warning("Attempting Gemini reconnect in %.1fs", delay_seconds)
        await asyncio.sleep(delay_seconds)

        await self._close_session()
        config = types.LiveConnectConfig(
            response_modalities=self._effective_response_modalities(),
            tools=[self._build_tool_declaration()],
            system_instruction=self.system_instruction,
        )
        try:
            await self._open_session(config)
        except Exception:  # noqa: BLE001
            logger.exception("Gemini reconnect attempt failed")
            return False

        logger.info("Gemini reconnect successful")
        return True

    async def _maybe_send_keepalive(self) -> None:
        if self.keepalive_seconds <= 0:
            return
        if not hasattr(self, "_last_keepalive_ts"):
            self._last_keepalive_ts = 0.0

        now = asyncio.get_event_loop().time()
        if now - self._last_keepalive_ts < float(self.keepalive_seconds):
            return

        self._last_keepalive_ts = now
        try:
            await self.send_user_text(self.keepalive_text)
            logger.info("Gemini keepalive prompt sent")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to send Gemini keepalive prompt")

    async def _start_realtime_inputs(self, session: Any) -> None:
        await self._stop_realtime_inputs()

        if self.audio_enabled:
            self._start_audio_capture(session)

        if self.video_enabled:
            self._start_video_capture(session)

    async def _stop_realtime_inputs(self) -> None:
        for task in self._media_tasks:
            task.cancel()
        for task in self._media_tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._media_tasks = []

        if self._audio_stream is not None:
            with suppress(Exception):
                self._audio_stream.stop()
                self._audio_stream.close()
            self._audio_stream = None

        self._audio_queue = None

    def _start_audio_capture(self, session: Any) -> None:
        if sd is None:
            logger.warning("sounddevice not available; skipping audio realtime input")
            return

        loop = asyncio.get_running_loop()
        self._audio_queue = asyncio.Queue(maxsize=64)

        def callback(indata: bytes, frames: int, _time_info: Any, status: Any) -> None:
            if status:
                logger.warning("Audio callback status: %s", status)
            if frames <= 0:
                return

            def enqueue() -> None:
                queue = self._audio_queue
                if queue is None:
                    return
                if queue.full():
                    with suppress(asyncio.QueueEmpty):
                        queue.get_nowait()
                queue.put_nowait(bytes(indata))

            loop.call_soon_threadsafe(enqueue)

        try:
            self._audio_stream = sd.RawInputStream(
                samplerate=self.audio_sample_rate_hz,
                blocksize=self.audio_blocksize_frames,
                dtype="int16",
                channels=1,
                callback=callback,
                device=self.audio_input_device,
            )
            self._audio_stream.start()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to start microphone capture")
            self._audio_stream = None
            self._audio_queue = None
            return

        task = asyncio.create_task(self._audio_sender_loop(session), name="gemini_audio_sender")
        task.add_done_callback(self._log_media_task_error)
        self._media_tasks.append(task)
        logger.info("Gemini audio capture started (device=%s)", self.audio_input_device)

    def _start_video_capture(self, session: Any) -> None:
        if cv2 is None:
            logger.warning("opencv-python not available; skipping video realtime input")
            return

        task = asyncio.create_task(self._video_sender_loop(session), name="gemini_video_sender")
        task.add_done_callback(self._log_media_task_error)
        self._media_tasks.append(task)
        if self.video_frame_provider is not None:
            logger.info("Gemini OBS video feed task started (program/source screenshots)")
        else:
            logger.info("Gemini camera video capture task started (device_index=%s)", self.video_device_index)

    async def _audio_sender_loop(self, session: Any) -> None:
        sent_chunks = 0
        while self._connected and self._session is session:
            queue = self._audio_queue
            if queue is None:
                return

            chunk = await queue.get()
            await session.send_realtime_input(
                audio=types.Blob(
                    data=chunk,
                    mime_type=f"audio/pcm;rate={self.audio_sample_rate_hz}",
                )
            )
            sent_chunks += 1
            if sent_chunks % 100 == 0:
                logger.info("Gemini audio sender heartbeat | chunks=%d", sent_chunks)

    async def _video_sender_loop(self, session: Any) -> None:
        if self.video_frame_provider is not None:
            await self._obs_frame_sender_loop(session)
            return

        if cv2 is None:
            return

        cap = cv2.VideoCapture(self.video_device_index)
        if not cap.isOpened():
            logger.warning("Failed to open video capture device index=%s", self.video_device_index)
            return

        with suppress(Exception):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.video_width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.video_height))

        interval = 1.0 / self.video_fps
        sent_frames = 0

        try:
            while self._connected and self._session is session:
                ok, frame = await asyncio.to_thread(cap.read)
                if not ok:
                    await asyncio.sleep(0.5)
                    continue

                encode_ok, encoded = await asyncio.to_thread(
                    cv2.imencode,
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.video_jpeg_quality],
                )
                if not encode_ok:
                    await asyncio.sleep(interval)
                    continue

                await session.send_realtime_input(
                    video=types.Blob(
                        data=encoded.tobytes(),
                        mime_type="image/jpeg",
                    )
                )

                sent_frames += 1
                if sent_frames % 25 == 0:
                    logger.info("Gemini video sender heartbeat | frames=%d", sent_frames)

                await asyncio.sleep(interval)
        finally:
            with suppress(Exception):
                cap.release()

    async def _obs_frame_sender_loop(self, session: Any) -> None:
        interval = 1.0 / self.video_fps
        sent_frames = 0

        while self._connected and self._session is session:
            try:
                frame_bytes = await self.video_frame_provider()
            except Exception:  # noqa: BLE001
                logger.exception("OBS frame provider failed")
                await asyncio.sleep(interval)
                continue

            if not frame_bytes:
                await asyncio.sleep(interval)
                continue

            await session.send_realtime_input(
                video=types.Blob(
                    data=frame_bytes,
                    mime_type="image/jpeg",
                )
            )

            sent_frames += 1
            if sent_frames % 25 == 0:
                logger.info("Gemini OBS video sender heartbeat | frames=%d", sent_frames)

            await asyncio.sleep(interval)

    @staticmethod
    def _parse_audio_device(value: str | None) -> int | str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned.isdigit():
            return int(cleaned)
        return cleaned

    def _require_session(self) -> Any:
        if not self._connected or self._session is None:
            raise RuntimeError("Gemini bridge not connected")
        return self._session

    @staticmethod
    def _log_media_task_error(task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001
            logger.exception("Gemini media task failed: %s", task.get_name())
