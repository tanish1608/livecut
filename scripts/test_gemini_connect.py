from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

sys.path.insert(0, "src")

from livecut.config import settings
from livecut.gemini_bridge import GeminiLiveBridge


async def main() -> None:
    load_dotenv()

    bridge = GeminiLiveBridge(
        model=settings.live_model,
        tool_schemas=[
            {
                "name": "switch_scene",
                "description": "Switch OBS program scene",
                "parameters": {
                    "type": "object",
                    "properties": {"scene_name": {"type": "string"}},
                    "required": ["scene_name"],
                },
            }
        ],
        use_vertexai=settings.google_genai_use_vertexai,
        api_key=settings.google_api_key,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        system_instruction="LiveCut connection smoke test",
    )

    await bridge.connect()
    print("LIVE_CONNECT_OK")
    await bridge.disconnect()
    print("LIVE_DISCONNECT_OK")


if __name__ == "__main__":
    asyncio.run(main())
