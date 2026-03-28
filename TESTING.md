# LiveCut Testing Guide

This document explains how to validate the current LiveCut prototype end to end.

## Scope

Covers:
- Python runtime startup and shutdown
- OBS WebSocket connectivity
- Simulation-mode event handling (`ENABLE_GEMINI=false`)
- Feature behavior mapping (scene switching, mute, overlays, chat highlight, segment timer)
- Gemini-mode startup path (`ENABLE_GEMINI=true`)

Does not yet cover:
- Real Gemini audio/video ingestion and tool call streaming (bridge is a scaffold)

## Prerequisites

1. Python 3.11+ installed.
2. OBS Studio 28+ with WebSocket enabled.
3. At least these OBS objects created:
- Scenes: `Gameplay_Focus`, `Chatting_Focus`
- Audio input: `Host Mic`
- Text sources: `LowerThirdText`, `HostPromptText`, `ChatQuestionText`
- Media source: `SFX_Airhorn`
- Optional image source for B-roll testing

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create environment file:

```bash
cp .env.example .env
```

3. Configure `.env`:
- `OBS_HOST`, `OBS_PORT`, `OBS_PASSWORD`
- `SCENE_GAMEPLAY_FOCUS`, `SCENE_CHATTING_FOCUS`
- `INPUT_HOST_MIC`
- `SOURCE_SFX_AIRHORN`, `SOURCE_LOWER_THIRD_TEXT`, `SOURCE_HOST_PROMPT_TEXT`, `SOURCE_CHAT_QUESTION_TEXT`
- `SEGMENT_MAX_MINUTES=1` (recommended for fast timer testing)
- `CHAT_BATCH_SECONDS=10` (recommended for fast chat testing)
- `ENABLE_GEMINI=false`

## Test 1: Startup Smoke Test

Command:

```bash
PYTHONPATH=src python -m livecut.main
```

Expected:
- Process starts and stays alive.
- Logs show OBS connection success.
- No uncaught exceptions.

Pass criteria:
- Runtime remains healthy for at least 60 seconds.

## Test 2: Scene Routing (Simulation Vision Loop)

While runtime is running, observe program scene in OBS for ~20 seconds.

Expected:
- Scene toggles between `Gameplay_Focus` and `Chatting_Focus` as simulated state changes.

Pass criteria:
- Multiple scene transitions occur without errors.

## Test 3: Smart Cough/Mute Behavior (Simulation Audio Loop)

While runtime is running, monitor `Host Mic` mute state.

Expected:
- Intermittent transient events trigger momentary mute.
- Mic unmutes automatically after `COUGH_RECOVERY_SECONDS`.

Pass criteria:
- Mute operation recovers every time and does not stick muted.

## Test 4: Killfeed SFX Trigger

While runtime is running, watch for random killfeed detections in logs.

Expected:
- When simulated killfeed includes `HOST_USERNAME`, `SFX_Airhorn` is triggered.

Pass criteria:
- Media source restart is invoked on matching events.

## Test 5: Chat Highlighting

With `CHAT_BATCH_SECONDS=10`, wait for at least 2 batches.

Expected:
- `ChatQuestionText` updates periodically with selected question text.

Pass criteria:
- Text source changes at each batch interval.

## Test 6: Segment Timer Prompt

Set `SEGMENT_MAX_MINUTES=1` and restart.

Expected:
- At 1 minute, `HostPromptText` updates with topic timeout reminder.

Pass criteria:
- Prompt appears on schedule and repeats every interval.

## Test 7: Clean Shutdown

Press `Ctrl+C`.

Expected:
- Runtime stops cleanly.
- OBS WebSocket disconnect log appears.
- Process exits with no hanging tasks.

Pass criteria:
- Exit returns control to shell quickly.

## Test 8: Gemini Mode Bootstrap

1. Set `ENABLE_GEMINI=true`.
2. Set model/env credentials as needed.
3. Run the app again.

Expected:
- Runtime initializes `GeminiLiveBridge`.
- No crash during bridge connect/disconnect path.

Pass criteria:
- App starts and stays alive with bridge enabled.

Note:
- Real multimodal streaming events are not produced yet until `GeminiLiveBridge.signals()` is implemented.

## Troubleshooting Checklist

- `Connection refused` to OBS:
  - Confirm OBS is running.
  - Confirm WebSocket server is enabled and port/password match `.env`.
- `Source not found` or `Scene not found`:
  - Verify exact OBS names match this document.
- Runtime starts but no visible actions:
  - Ensure required scenes/sources exist in current scene collection.
  - Lower `CHAT_BATCH_SECONDS` and `SEGMENT_MAX_MINUTES` for faster feedback.

## Optional Repeatable Verification Script

Use this sequence each time you change runtime/tool logic:

1. Run startup smoke test.
2. Observe at least one scene switch.
3. Observe at least one chat text update.
4. Observe at least one timer prompt.
5. Perform clean shutdown.

If all five pass, the core control loop is considered healthy.
