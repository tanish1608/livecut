# LiveCut: The Multimodal AI Producer

LiveCut is a low-latency Python middleware that turns multimodal AI decisions into real OBS Studio actions.

## What this starter includes

- OBS WebSocket control layer for scene switching, muting, overlays, text, and SFX
- Tool registry to map AI function calls into production-safe actions
- Async signal loops for:
  - visual pacing and killfeed reactions
  - audio transient handling (cough/sneeze style auto-mute)
  - segment timeout prompts
  - periodic chat question highlighting
- Gemini live bridge with reconnects, keepalive prompts, and realtime media ingestion (mic + optional camera)
- Optional NVIDIA Nemotron VLM bridge for 1s frame-based vision decisions

## Role Split (Recommended)

- NVIDIA VLM = live director and vision context feed (plus kill detection)
- Gemini Live = execution agent for tool calls (voice commands + selective autonomous actions)
- OBS tools = final action layer with safety guards

This mode keeps direction and execution separate so you can talk to the director while Gemini handles concrete OBS operations.

## Simple Voice Assistant Mode (Recommended for reliability)

This mode removes the continuous Gemini Live stream loop and uses a simpler architecture:

- Chrome Web Speech hears your voice.
- Wake word detection happens in the browser (say: `gemini ...`).
- The command text is sent to Gemini command model.
- LiveCut executes one OBS tool call and speaks confirmation using macOS `say`.

Enable in `.env`:

- `GEMINI_SIMPLE_ASSISTANT_MODE=true`
- `GEMINI_COMMAND_MODEL=gemini-2.5-flash`
- `CHROME_LISTENER_HOST=127.0.0.1`
- `CHROME_LISTENER_PORT=8765`
- `CHROME_AUTO_OPEN=true`
- `GEMINI_VOICE_WAKE_WORD=gemini`
- `GEMINI_SPEAK_REPLIES=true`

For simple assistant mode, keep these disabled to reduce noise:

- `ENABLE_VLM=false`
- `GEMINI_VIDEO_ENABLED=false`
- `RUN_SIMULATION_LOOPS_WITH_GEMINI=false`

Voice commands now support broader OBS control, including:

- scene switches: `switch_scene`
- text overlays: `show_lower_third`, `clear_lower_third`, `show_host_prompt`, `clear_host_prompt`, `highlight_question`, `clear_chat_question`
- source visibility in current scene: `show_source_current_scene`, `hide_source_current_scene`
- audio controls: `mute_input`, `unmute_input`, `momentary_mute`, `play_sfx`
- b-roll image injection: `inject_broll_from_url`

Examples:

- "Gemini, show lower third saying welcome everyone"
- "Gemini, highlight question what sensitivity do you use"
- "Gemini, hide source webcam"
- "Gemini, switch to intro scene and show host prompt we're starting now"

## Project layout

- `src/livecut/obs_controller.py`: OBS command execution (obsws-python)
- `src/livecut/tools.py`: callable production tools and JSON-like schemas
- `src/livecut/agent_runtime.py`: event router and orchestration
- `src/livecut/signal_loops.py`: simulated vision/audio/chat/timer streams for local testing
- `src/livecut/gemini_bridge.py`: integration boundary for Gemini Multimodal Live
- `src/livecut/main.py`: app entrypoint

## Quickstart

1. Create and activate a virtual env.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure env:

```bash
cp .env.example .env
```

4. Set OBS WebSocket values in `.env` and start OBS with WebSocket enabled.
5. Run:

```bash
PYTHONPATH=src python -m livecut.main
```

If your OBS scene/source names differ, set these in `.env` to match your exact names:

- `SCENE_GAMEPLAY_FOCUS`
- `SCENE_CHATTING_FOCUS`
- `INPUT_HOST_MIC`
- `SOURCE_SFX_AIRHORN`
- `SOURCE_LOWER_THIRD_TEXT`
- `SOURCE_HOST_PROMPT_TEXT`
- `SOURCE_CHAT_QUESTION_TEXT`

## Current behavior

- With `ENABLE_GEMINI=false` (default), LiveCut runs in simulation mode and still controls OBS in real time.
- With `ENABLE_GEMINI=true`, runtime initializes `GeminiLiveBridge` and streams:
  - microphone PCM chunks to Gemini Live (`GEMINI_AUDIO_ENABLED=true`)
  - optional camera frames as JPEG (`GEMINI_VIDEO_ENABLED=true`)
  - tool-call extraction and execution via `ToolRegistry`
  - context injection messages (segment warnings, chat prompts)
- Gemini mode now includes:
  - receive idle diagnostics and runtime heartbeat logs
  - optional keepalive text prompts (`GEMINI_KEEPALIVE_SECONDS`)
  - automatic reconnect when the live receive stream ends (`GEMINI_AUTO_RECONNECT`)
  - optional bootstrap user turn on connect (`GEMINI_BOOTSTRAP_USER_TEXT`)
  - optional hybrid mode (`RUN_SIMULATION_LOOPS_WITH_GEMINI=true`) to keep local signal loops active alongside Gemini
  - realtime microphone streaming to Gemini (`GEMINI_AUDIO_ENABLED=true`)
  - OBS program scene frame streaming at low FPS (default video mode)
  - optional direct camera-device mode (`GEMINI_VIDEO_SOURCE_MODE=camera_device`)
  - startup control-room context injection so model uses exact OBS scene/source names
- With `ENABLE_VLM=true`, runtime initializes `NvidiaVLMBridge` and:
  - captures OBS screenshots on a fixed interval (`VLM_POLL_SECONDS`, default 1.0s)
  - calls NVIDIA OpenAI-compatible endpoint (`VLM_BASE_URL`)
  - parses structured action JSON and emits tool calls into the same safe `ToolRegistry`
  - can run in parallel with Gemini so vision and audio are split across models

## Feature mapping from your design

- Contextual PiP / visual pacing: `agent_runtime.py` dispatch on `vision.frame_analysis`
- Smart cough/mute: `agent_runtime.py` + `tools.momentary_mute`
- Dynamic lower thirds and overlays: `tools.show_lower_third` + `tools.toggle_overlay`
- B-roll injection: `tools.inject_broll_from_url`
- Kill detection SFX: `agent_runtime.py` killfeed check + `tools.play_sfx`
- Chat highlighting: `chat_batch_loop` + `_pick_question` + `tools.highlight_question`
- Segment timers: `segment_timer_loop` + `show_lower_third` host prompt source

## Realtime Input Controls

Configure these in `.env` for Gemini streaming:

- `GEMINI_AUDIO_ENABLED=true|false`
- `GEMINI_VIDEO_ENABLED=true|false`
- `GEMINI_AUDIO_INPUT_DEVICE` (optional device index or name)
- `GEMINI_VIDEO_DEVICE_INDEX`
- `GEMINI_VIDEO_SOURCE_MODE` (`obs_program_scene` or `camera_device`)
- `GEMINI_VIDEO_SOURCE_NAME` (optional; defaults to current OBS program scene)
- `GEMINI_VIDEO_FPS`
- `GEMINI_VIDEO_WIDTH`
- `GEMINI_VIDEO_HEIGHT`
- `GEMINI_VIDEO_JPEG_QUALITY`
- `AI_ONLY_SCENE_SWITCHING` (`true` disables local vision scene heuristics so only Gemini tool calls can switch scenes)
- `GEMINI_SCENE_SWITCH_DELAY_SECONDS` (delay before executing Gemini `switch_scene`; use 2-6s to give AI more scene context)
- `SCENE_MIN_DWELL_SECONDS` (minimum time between scene switches to avoid thrash)

Voice assistant command mode:

- `GEMINI_VOICE_ASSISTANT_MODE=true` (opt-in voice-command behavior)
- `GEMINI_REQUIRE_WAKE_WORD=true` (execute only when wake word is present)
- `GEMINI_VOICE_WAKE_WORD=gemini` (example command: "Gemini, switch to chatting scene")
- `GEMINI_WAKE_WINDOW_SECONDS=8` (tool calls allowed only shortly after wake word is heard)
- `LIVE_RESPONSE_MODALITIES=` (leave empty to auto-pick; native-audio models will use `AUDIO`)
- `GEMINI_AUDIO_OUTPUT_ENABLED=true` (play Gemini voice replies on your speaker)
- `GEMINI_AUDIO_OUTPUT_DEVICE=` (optional output device index/name)

Minimal personal-assistant setup (simple mode):

- `ENABLE_GEMINI=true`
- `ENABLE_VLM=false`
- `GEMINI_VIDEO_ENABLED=false`
- `RUN_SIMULATION_LOOPS_WITH_GEMINI=false`
- `GEMINI_USE_VLM_CONTEXT=false`
- Speak commands like: "Gemini, switch to chatting scene" or "Gemini, show this image on stream ..."

NVIDIA VLM controls:

- `ENABLE_VLM=true|false`
- `VLM_MODEL` (example: `nvidia/nemotron-nano-12b-v2-vl`)
- `VLM_BASE_URL` (example: `https://integrate.api.nvidia.com`)
- `NVIDIA_API_KEY`
- `VLM_POLL_SECONDS` (set `1.0` for per-second frame decisions)
- `VLM_REQUEST_TIMEOUT_SECONDS`
- `VLM_MAX_ACTIONS_PER_TURN`
- `VLM_ALLOWED_TOOLS` (comma-separated allowlist, recommended: `switch_scene`)
- `VLM_ACTION_COOLDOWN_SECONDS` (dedupe repeat actions, recommended: `3-8`)
- `VLM_SCENE_SWITCH_DELAY_SECONDS` (separate delay for VLM scene switches; use `0.8-2.0` for low-latency)
- `VLM_ERROR_BACKOFF_BASE_SECONDS` (backoff on transient VLM 5xx errors)
- `VLM_ERROR_BACKOFF_MAX_SECONDS` (maximum 5xx backoff cap)
- `VLM_SYSTEM_INSTRUCTION` (optional override)

Director mode controls:

- `VLM_ROLE=director`
- `VLM_ENABLE_TOOL_CALLS=false` (recommended; VLM emits context, not direct tool calls)
- `VLM_KILL_DETECTION_ENABLED=true`
- `VLM_KILL_KEYWORDS=eliminated,knocked,kill,frag,headshot`

Gemini execution-from-context controls:

- `GEMINI_USE_VLM_CONTEXT=true`
- `GEMINI_VLM_CONTEXT_MIN_INTERVAL_SECONDS=2.0`
- `GEMINI_CHAT_ACTIONS_ONLY_IN_CHAT_SCENE=true`

Image/b-roll controls:

- `SOURCE_BROLL_IMAGE` (default target for `inject_broll_from_url` so voice commands can pull up images)

## Debugging Checklist

1. Confirm stream ingestion logs:
  - `Gemini audio capture started`
  - `Gemini audio sender heartbeat`
  - `Gemini video sender heartbeat` (if video enabled)
2. Confirm model events:
  - `Gemini tool-call event with ...`
  - `Gemini assistant transcript: ...`
3. Confirm execution:
  - `Executing Gemini tool call: ...`
  - OBS action logs (`Switched scene`, `Updated text source`, etc.)
  - `Applying Gemini scene switch delay of ...s before switch_scene` (when Gemini requests a scene switch)

## Testing

Use [TESTING.md](TESTING.md) for a complete step-by-step validation checklist.
