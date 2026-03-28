from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from .types import StreamSignal

logger = logging.getLogger(__name__)


class NvidiaVLMBridge:
    """Polling bridge for NVIDIA-hosted vision-language models.

    The bridge captures a frame on a fixed interval and emits structured
    director context for downstream routing. Optional tool-call passthrough
    can be enabled for advanced modes.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None,
        frame_provider: Callable[[], Awaitable[bytes | None]],
        tool_schemas: list[dict[str, Any]],
        scene_gameplay_focus: str,
        scene_chatting_focus: str,
        source_sfx_airhorn: str,
        source_host_prompt_text: str,
        source_chat_question_text: str,
        poll_seconds: float = 1.0,
        request_timeout_seconds: float = 12.0,
        max_actions_per_turn: int = 2,
        allowed_tool_names: list[str] | None = None,
        action_cooldown_seconds: float = 5.0,
        error_backoff_base_seconds: float = 1.0,
        error_backoff_max_seconds: float = 8.0,
        role: str = "director",
        enable_tool_calls: bool = False,
        kill_detection_enabled: bool = True,
        kill_keywords: list[str] | None = None,
        system_instruction: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.frame_provider = frame_provider
        self.tool_schemas = tool_schemas
        self.scene_gameplay_focus = scene_gameplay_focus
        self.scene_chatting_focus = scene_chatting_focus
        self.source_sfx_airhorn = source_sfx_airhorn
        self.source_host_prompt_text = source_host_prompt_text
        self.source_chat_question_text = source_chat_question_text
        self.poll_seconds = max(0.25, float(poll_seconds))
        self.request_timeout_seconds = max(2.0, float(request_timeout_seconds))
        self.max_actions_per_turn = max(1, int(max_actions_per_turn))
        self.action_cooldown_seconds = max(0.0, float(action_cooldown_seconds))
        self.error_backoff_base_seconds = max(0.25, float(error_backoff_base_seconds))
        self.error_backoff_max_seconds = max(self.error_backoff_base_seconds, float(error_backoff_max_seconds))
        self.role = role.strip().lower() if role else "director"
        self.enable_tool_calls = bool(enable_tool_calls)
        self.kill_detection_enabled = bool(kill_detection_enabled)
        self.kill_keywords = [k.strip().lower() for k in (kill_keywords or []) if isinstance(k, str) and k.strip()]
        self.system_instruction = system_instruction

        self._connected = False
        self._client: httpx.AsyncClient | None = None
        declared_tool_names = {
            str(schema.get("name"))
            for schema in self.tool_schemas
            if isinstance(schema.get("name"), str) and schema.get("name")
        }
        if allowed_tool_names:
            allow = {name for name in allowed_tool_names if isinstance(name, str) and name}
            self._tool_names = declared_tool_names.intersection(allow)
        else:
            self._tool_names = declared_tool_names
        self._turn_id = 0
        self._last_action_ts: dict[str, float] = {}
        self._error_backoff_seconds = self.error_backoff_base_seconds
        self._last_summary_text: str = ""

    async def connect(self) -> None:
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY is required when ENABLE_VLM=true")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.request_timeout_seconds,
        )
        self._connected = True
        logger.info(
            "NVIDIA VLM bridge initialized for model=%s base_url=%s poll=%.2fs",
            self.model,
            self.base_url,
            self.poll_seconds,
        )

    async def disconnect(self) -> None:
        self._connected = False
        if self._client is not None:
            await self._client.aclose()
        self._client = None

    async def signals(self) -> AsyncIterator[StreamSignal]:
        while self._connected:
            loop_start = asyncio.get_event_loop().time()

            frame_bytes = await self.frame_provider()
            if not frame_bytes:
                await asyncio.sleep(self.poll_seconds)
                continue

            try:
                director = await self._infer_director_context(frame_bytes)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status >= 500:
                    logger.warning(
                        "NVIDIA VLM upstream error status=%s. Backing off for %.2fs",
                        status,
                        self._error_backoff_seconds,
                    )
                    await asyncio.sleep(self._error_backoff_seconds)
                    self._error_backoff_seconds = min(self._error_backoff_seconds * 2.0, self.error_backoff_max_seconds)
                else:
                    logger.error("NVIDIA VLM request failed status=%s", status)
                    await asyncio.sleep(self.poll_seconds)
                continue
            except Exception:  # noqa: BLE001
                logger.exception("NVIDIA VLM inference failed")
                await asyncio.sleep(self.poll_seconds)
                continue

            self._error_backoff_seconds = self.error_backoff_base_seconds
            summary = str(director.get("summary", "")).strip()
            if summary and summary != self._last_summary_text:
                self._last_summary_text = summary
                yield StreamSignal(source="vlm", kind="assistant_transcript", payload={"text": summary})

            context_payload = {
                "summary": summary,
                "focus": str(director.get("focus", "neutral")),
                "kill_detected": bool(director.get("kill_detected", False)),
                "needs_gemini_action": bool(director.get("needs_gemini_action", False)),
                "requested_actions": director.get("requested_actions", []),
            }
            yield StreamSignal(source="vlm", kind="director_context", payload=context_payload)

            actions = director.get("actions", []) if self.enable_tool_calls else []
            if not isinstance(actions, list):
                actions = []

            for action in actions:
                if not self._is_action_allowed_now(action):
                    continue
                self._turn_id += 1
                yield StreamSignal(
                    source="vlm",
                    kind="tool_call",
                    payload={
                        "id": f"vlm-{self._turn_id}",
                        "name": action["name"],
                        "arguments": action["arguments"],
                    },
                )

            elapsed = asyncio.get_event_loop().time() - loop_start
            sleep_seconds = max(0.0, self.poll_seconds - elapsed)
            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)

    async def _infer_director_context(self, frame_bytes: bytes) -> dict[str, Any]:
        client = self._require_client()
        frame_b64 = base64.b64encode(frame_bytes).decode("ascii")

        tool_names = sorted(self._tool_names)
        system_text = self.system_instruction or self._default_system_instruction()
        user_text = (
            "Analyze the frame and return strict JSON with fields: "
            "summary (string), focus (gameplay|chatting|neutral), "
            "kill_detected (boolean), needs_gemini_action (boolean), "
            "requested_actions (array of short action intents), actions (array). "
            "Each action item must be {name: string, arguments: object}. "
            "If no action is needed, return actions as an empty array. "
            f"Allowed tools: {tool_names}. "
            f"Gameplay scene: {self.scene_gameplay_focus}. "
            f"Chatting scene: {self.scene_chatting_focus}. "
            f"SFX source: {self.source_sfx_airhorn}. "
            f"Host prompt text source: {self.source_host_prompt_text}. "
            f"Chat question text source: {self.source_chat_question_text}."
        )

        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 500,
            "messages": [
                {"role": "system", "content": system_text},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                    ],
                },
            ],
        }

        response = await client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        content = self._extract_message_content(data)
        parsed = self._parse_json_payload(content)

        summary = str(parsed.get("summary", "")).strip()
        focus = str(parsed.get("focus", "neutral")).strip().lower()
        if focus not in {"gameplay", "chatting", "neutral"}:
            focus = "neutral"

        requested_actions = parsed.get("requested_actions", [])
        if not isinstance(requested_actions, list):
            requested_actions = []

        kill_detected = bool(parsed.get("kill_detected", False))
        if self.kill_detection_enabled and not kill_detected and summary:
            s = summary.lower()
            kill_detected = any(k in s for k in self.kill_keywords)

        needs_gemini_action = bool(parsed.get("needs_gemini_action", False))
        if focus == "chatting":
            needs_gemini_action = True

        raw_actions = parsed.get("actions", [])
        if not isinstance(raw_actions, list):
            raw_actions = []

        actions: list[dict[str, Any]] = []
        switch_scene_emitted = False
        for item in raw_actions:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            arguments = item.get("arguments", {})
            if not isinstance(name, str) or name not in self._tool_names:
                continue
            if not isinstance(arguments, dict):
                arguments = {}
            normalized = self._normalize_action(name=name, arguments=arguments)
            if normalized is None:
                continue
            if normalized["name"] == "switch_scene":
                if switch_scene_emitted:
                    continue
                switch_scene_emitted = True
            actions.append(normalized)
            if len(actions) >= self.max_actions_per_turn:
                break

        return {
            "summary": summary,
            "focus": focus,
            "kill_detected": kill_detected,
            "needs_gemini_action": needs_gemini_action,
            "requested_actions": [str(x) for x in requested_actions[:6]],
            "actions": actions,
        }

    def _default_system_instruction(self) -> str:
        if self.role == "director":
            return (
                "You are the live director. Describe what is happening and emit concise director context. "
                "Prioritize kill detection and whether Gemini should act. "
                "Do not output OBS tool actions unless explicitly required."
            )
        return (
            "You are a live stream producer. Decide only the minimum safe OBS tool actions from visuals. "
            "Never invent scene/source names. Avoid rapid scene thrashing."
        )

    def _normalize_action(self, name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        if name != "switch_scene":
            return {"name": name, "arguments": arguments}

        scene_candidate = (
            arguments.get("scene_name")
            or arguments.get("scene")
            or arguments.get("target")
            or arguments.get("target_scene")
            or arguments.get("sceneName")
        )
        if not isinstance(scene_candidate, str) or not scene_candidate.strip():
            return None

        resolved = self._resolve_scene_name(scene_candidate)
        if not resolved:
            return None

        return {"name": name, "arguments": {"scene_name": resolved}}

    def _resolve_scene_name(self, candidate: str) -> str | None:
        cleaned = candidate.strip()
        if not cleaned:
            return None

        by_exact = {
            self.scene_gameplay_focus.lower(): self.scene_gameplay_focus,
            self.scene_chatting_focus.lower(): self.scene_chatting_focus,
        }
        if cleaned.lower() in by_exact:
            return by_exact[cleaned.lower()]

        lowered = cleaned.lower()
        if any(token in lowered for token in ("game", "combat", "fight", "action")):
            return self.scene_gameplay_focus
        if any(token in lowered for token in ("chat", "talk", "menu", "intermission")):
            return self.scene_chatting_focus
        return None

    def _is_action_allowed_now(self, action: dict[str, Any]) -> bool:
        if self.action_cooldown_seconds <= 0:
            return True

        key = json.dumps(action, sort_keys=True, separators=(",", ":"))
        now = asyncio.get_event_loop().time()
        prev = self._last_action_ts.get(key)
        if prev is not None and (now - prev) < self.action_cooldown_seconds:
            return False
        self._last_action_ts[key] = now
        return True

    @staticmethod
    def _extract_message_content(data: dict[str, Any]) -> str:
        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            return "\n".join(parts)

        return ""

    @staticmethod
    def _parse_json_payload(content: str) -> dict[str, Any]:
        text = content.strip()
        if not text:
            return {}

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            snippet = text[start : end + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}
        return {}

    def _require_client(self) -> httpx.AsyncClient:
        if not self._connected or self._client is None:
            raise RuntimeError("NVIDIA VLM bridge not connected")
        return self._client
