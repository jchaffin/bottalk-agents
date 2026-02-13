"""
Sarah — sales rep participant.

    python sarah.py --room-url https://… --token … --system-prompt "…"
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"), override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG", filter=lambda r: "rtvi" not in r["name"])

VOICE_ID = os.getenv("SARAH_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")


async def main(room_url: str, token: str, system_prompt: str):
    from agent import run_agent

    logger.info(f"[Sarah] joining {room_url}")
    await run_agent(
        room_url=room_url,
        token=token,
        name="Sarah",
        system_prompt=system_prompt,
        voice_id=VOICE_ID,
        goes_first=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--room-url", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--system-prompt", required=True, help="System prompt for Sarah")
    args = p.parse_args()
    try:
        asyncio.run(main(args.room_url, args.token, args.system_prompt))
    except KeyboardInterrupt:
        logger.info("[Sarah] stopped.")
