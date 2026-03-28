from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

from google import genai

from .tools import ToolRegistry

logger = logging.getLogger(__name__)


_HTML_PAGE = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>LiveCut Chrome Wake Listener</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; }
    button { font-size: 16px; padding: 10px 14px; }
    #log { margin-top: 12px; white-space: pre-wrap; border: 1px solid #ddd; padding: 10px; height: 220px; overflow: auto; }
  </style>
</head>
<body>
  <h2>LiveCut Wake Listener</h2>
  <p>Click Start and say commands like: <b>Gemini, switch to chatting scene</b>.</p>
  <button id=\"toggle\">Start Listening</button>
  <div id=\"log\"></div>
  <script>
    const log = (t) => {
      const el = document.getElementById('log');
      el.textContent += `${new Date().toLocaleTimeString()} ${t}\n`;
      el.scrollTop = el.scrollHeight;
    };

    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      log('SpeechRecognition not available. Open this page in Chrome.');
    }

    let rec = null;
    let running = false;

    function postText(text) {
      return fetch('/transcript', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
    }

    function start() {
      if (!SR) return;
      rec = new SR();
      rec.continuous = true;
      rec.interimResults = false;
      rec.lang = 'en-US';

      rec.onresult = async (event) => {
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const r = event.results[i];
          if (!r.isFinal) continue;
          const text = r[0].transcript.trim();
          if (!text) continue;
          log(`heard: ${text}`);
          try {
            await postText(text);
          } catch (e) {
            log('failed to post transcript');
          }
        }
      };

      rec.onerror = (e) => {
        log(`error: ${e.error}`);
      };

      rec.onend = () => {
        if (running) {
          try { rec.start(); } catch (_) {}
        }
      };

      running = true;
      rec.start();
      log('listening started');
    }

    function stop() {
      running = false;
      if (rec) {
        rec.onend = null;
        rec.stop();
      }
      log('listening stopped');
    }

    document.getElementById('toggle').onclick = () => {
      if (!running) {
        start();
        document.getElementById('toggle').textContent = 'Stop Listening';
      } else {
        stop();
        document.getElementById('toggle').textContent = 'Start Listening';
      }
    };
  </script>
</body>
</html>
"""


class _ChromeSpeechHandler(BaseHTTPRequestHandler):
    queue: asyncio.Queue[str] | None = None
    loop: asyncio.AbstractEventLoop | None = None

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/":
            self.send_response(404)
            self.end_headers()
            return
        payload = _HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/transcript":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            self.send_response(400)
            self.end_headers()
            return

        text = str(data.get("text", "")).strip()
        if text and self.queue is not None and self.loop is not None:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, text)

        self.send_response(204)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


class ChromeWakeListener:
    def __init__(self, host: str, port: int, auto_open: bool) -> None:
        self.host = host
        self.port = port
        self.auto_open = auto_open
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=128)
        self._server: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        _ChromeSpeechHandler.queue = self._queue
        _ChromeSpeechHandler.loop = loop
        self._server = ThreadingHTTPServer((self.host, self.port), _ChromeSpeechHandler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        url = f"http://{self.host}:{self.port}/"
        logger.info("Chrome wake listener started at %s", url)
        if self.auto_open:
            webbrowser.open(url)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._thread = None

    async def next_transcript(self) -> str:
        return await self._queue.get()


@dataclass
class AssistantConfig:
    wake_word: str
    command_model: str
    speak_replies: bool


class SimpleVoiceAssistant:
    def __init__(
        self,
        tools: ToolRegistry,
        listener: ChromeWakeListener,
        config: AssistantConfig,
        client: genai.Client,
        scene_gameplay_focus: str,
        scene_chatting_focus: str,
        source_sfx_airhorn: str,
        source_host_prompt_text: str,
        source_chat_question_text: str,
        source_broll_image: str,
    ) -> None:
        self.tools = tools
        self.listener = listener
        self.config = config
        self.client = client
        self.scene_gameplay_focus = scene_gameplay_focus
        self.scene_chatting_focus = scene_chatting_focus
        self.source_sfx_airhorn = source_sfx_airhorn
        self.source_host_prompt_text = source_host_prompt_text
        self.source_chat_question_text = source_chat_question_text
        self.source_broll_image = source_broll_image

    async def run(self) -> None:
        await self.listener.start()
        try:
            await self._speak("Voice assistant ready. Say Gemini followed by your command.")
            while True:
                transcript = await self.listener.next_transcript()
                await self._handle_transcript(transcript)
        finally:
            await self.listener.stop()

    async def _handle_transcript(self, transcript: str) -> None:
        cleaned = transcript.strip()
        if not cleaned:
            return

        lower = cleaned.lower()
        wake = self.config.wake_word.lower()
        if wake not in lower:
            return

        command = cleaned
        wake_idx = lower.find(wake)
        if wake_idx >= 0:
            command = cleaned[wake_idx + len(wake) :].strip(" ,:.-")

        if not command:
            await self._speak("Yes?")
            return

        logger.info("Wake command: %s", command)
        decision = await self._plan_tool_call(command)

        tool_name = str(decision.get("tool_name", "")).strip()
        args = decision.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        speak_text = str(decision.get("speak", "")).strip()

        if tool_name:
            try:
                result = await self.tools.execute(tool_name, args)
                logger.info("Assistant executed tool=%s result=%s", tool_name, result)
                if not speak_text:
                    speak_text = f"Done. {tool_name.replace('_', ' ')} complete."
            except Exception as exc:  # noqa: BLE001
                logger.exception("Assistant tool execution failed")
                speak_text = f"I could not execute {tool_name}. {exc}"
        elif not speak_text:
            speak_text = "I heard you. Please give me a specific stream command."

        await self._speak(speak_text)

    async def _plan_tool_call(self, command: str) -> dict[str, Any]:
        tool_names = [schema.get("name") for schema in self.tools.tool_schemas if isinstance(schema.get("name"), str)]
        schema_hint = {
            "tool_name": "string or empty",
            "arguments": "object",
            "speak": "short response to user",
        }
        prompt = (
            "You are a command router for livestream control.\n"
            "Return strict JSON only. No markdown.\n"
            f"JSON schema: {json.dumps(schema_hint)}\n"
            f"Allowed tool names: {tool_names}\n"
            "Use these exact names when needed:\n"
            f"- gameplay scene: {self.scene_gameplay_focus}\n"
            f"- chatting scene: {self.scene_chatting_focus}\n"
            f"- airhorn source: {self.source_sfx_airhorn}\n"
            f"- host prompt source: {self.source_host_prompt_text}\n"
            f"- chat question source: {self.source_chat_question_text}\n"
            f"- broll image source: {self.source_broll_image}\n"
            "If no tool is needed, set tool_name to empty string and provide speak.\n"
            f"User command: {command}"
        )

        def _request() -> str:
            response = self.client.models.generate_content(model=self.config.command_model, contents=prompt)
            text = getattr(response, "text", None)
            return text or ""

        raw = await asyncio.to_thread(_request)
        parsed = self._parse_json(raw)
        if parsed is None:
            return {"tool_name": "", "arguments": {}, "speak": raw.strip() or "Sorry, I did not understand."}
        return parsed

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        candidate = text.strip()
        if not candidate:
            return None
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    async def _speak(self, text: str) -> None:
        if not text:
            return
        logger.info("Assistant reply: %s", text)
        if not self.config.speak_replies:
            return

        def _say() -> None:
            subprocess.run(["say", text], check=False)

        await asyncio.to_thread(_say)
