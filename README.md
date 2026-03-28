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
- Gemini bridge boundary (`GeminiLiveBridge`) so you can wire the exact Gemini Live SDK flow without refactoring runtime logic

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
- With `ENABLE_GEMINI=true`, runtime initializes `GeminiLiveBridge`. The bridge file is where to plug in:
  - camera/audio frame ingestion
  - tool-call extraction from Gemini responses
  - context injection messages (segment warnings, etc.)

## Feature mapping from your design

- Contextual PiP / visual pacing: `agent_runtime.py` dispatch on `vision.frame_analysis`
- Smart cough/mute: `agent_runtime.py` + `tools.momentary_mute`
- Dynamic lower thirds and overlays: `tools.show_lower_third` + `tools.toggle_overlay`
- B-roll injection: `tools.inject_broll_from_url`
- Kill detection SFX: `agent_runtime.py` killfeed check + `tools.play_sfx`
- Chat highlighting: `chat_batch_loop` + `_pick_question` + `tools.highlight_question`
- Segment timers: `segment_timer_loop` + `show_lower_third` host prompt source

## Next integration step (Gemini Live)

Implement these inside `GeminiLiveBridge`:

- persistent live session setup
- frame/audio streaming to Gemini
- streaming receive loop that yields `StreamSignal` objects
- function-call passthrough to `ToolRegistry.execute`

This separation keeps the hard real-time OBS control logic stable while you iterate on prompt and model behavior.

## Testing

Use [TESTING.md](TESTING.md) for a complete step-by-step validation checklist.
