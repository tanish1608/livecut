from __future__ import annotations

import logging
from contextlib import suppress
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import types
from google.auth.exceptions import DefaultCredentialsError

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
    ) -> None:
        self.model = model
        self.tool_schemas = tool_schemas
        self.use_vertexai = use_vertexai
        self.api_key = api_key
        self.project = project
        self.location = location
        self.system_instruction = system_instruction
        self.response_modalities = response_modalities

        self._client: genai.Client | None = None
        self._live_cm: Any | None = None
        self._session: Any | None = None
        self._connected = False

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

        self._live_cm = self._client.aio.live.connect(model=self.model, config=config)
        try:
            self._session = await self._live_cm.__aenter__()
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
        logger.info("Gemini bridge initialized for model=%s", self.model)

    async def disconnect(self) -> None:
        if self._live_cm is not None:
            with suppress(Exception):
                await self._live_cm.__aexit__(None, None, None)
        if self._client is not None:
            self._client.close()

        self._live_cm = None
        self._session = None
        self._client = None
        self._connected = False

    async def inject_system_message(self, text: str) -> None:
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
        session = self._require_session()

        async for msg in session.receive():
            setup_complete = bool(getattr(msg, "setup_complete", False))
            if setup_complete:
                yield StreamSignal(source="gemini", kind="setup_complete", payload={})

            tool_call = getattr(msg, "tool_call", None)
            if tool_call is not None:
                function_calls = list(getattr(tool_call, "function_calls", []) or [])
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
                        yield StreamSignal(source="audio", kind="speech", payload={"speaker": "host", "text": text})

                output_tx = getattr(server_content, "output_transcription", None)
                if output_tx is not None:
                    text = getattr(output_tx, "text", None)
                    if text:
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

    def _require_session(self) -> Any:
        if not self._connected or self._session is None:
            raise RuntimeError("Gemini bridge not connected")
        return self._session
