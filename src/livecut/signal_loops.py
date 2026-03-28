from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator

from .types import StreamSignal

logger = logging.getLogger(__name__)


async def fake_vision_loop(host_username: str) -> AsyncIterator[StreamSignal]:
    """Simulates vision events until real Gemini video ingestion is wired."""
    scenes = ["chat", "combat", "menu"]
    while True:
        await asyncio.sleep(2.0)
        scene_state = random.choice(scenes)
        payload = {"state": scene_state}
        if scene_state == "combat" and random.random() > 0.7:
            payload["killfeed"] = f"{host_username} eliminated enemy42"
        yield StreamSignal(source="vision", kind="frame_analysis", payload=payload)


async def fake_audio_loop() -> AsyncIterator[StreamSignal]:
    """Simulates audio spikes and speech windows for local dev."""
    while True:
        await asyncio.sleep(1.5)
        roll = random.random()
        if roll > 0.92:
            yield StreamSignal(source="audio", kind="transient_spike", payload={"input_name": "Host Mic"})
        elif roll > 0.60:
            yield StreamSignal(source="audio", kind="speech", payload={"speaker": "host"})


async def segment_timer_loop(minutes: int) -> AsyncIterator[StreamSignal]:
    while True:
        await asyncio.sleep(minutes * 60)
        logger.info("Segment timer reached %d minute mark", minutes)
        yield StreamSignal(source="timer", kind="segment_timeout", payload={"minutes": minutes})


async def chat_batch_loop(seconds: int) -> AsyncIterator[StreamSignal]:
    """Stub loop that can be replaced with Twitch/YouTube polling."""
    sample_questions = [
        "What GPU settings are you using for this game?",
        "Can you share your current stream audio chain?",
        "Would you benchmark this setup against last week?",
    ]
    while True:
        await asyncio.sleep(seconds)
        yield StreamSignal(source="chat", kind="chat_batch", payload={"messages": sample_questions})
