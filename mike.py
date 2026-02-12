"""
Mike — potential customer participant.

    python mike.py --room-url https://… --token …
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

TOPIC = os.getenv("CONVERSATION_TOPIC", "enterprise software sales")

SYSTEM_PROMPT = f"""\
You are Mike, VP of Ops at BrightCart, a 200-person e-commerce company. \
A sales rep just called you about {TOPIC}.

Your pain: manual order processing, poor tool integration, team drowning \
in repetitive tasks. Budget ~$50k/yr. Last year you bought an expensive \
platform that flopped, so you are cautious.

Rules:
- 2-3 short spoken sentences per turn. No bullets, no markdown, no emoji.
- Be interested but skeptical. Push back on price. Ask for proof of ROI.
- Do not agree too quickly. Ask pointed questions about timeline and support."""

VOICE_ID = os.getenv("MIKE_VOICE_ID", "TxGEqnHWrfWFTfGW9XjX")


async def main(room_url: str, token: str):
    from agent import run_agent

    logger.info(f"[Mike] joining {room_url}")
    await run_agent(
        room_url=room_url,
        token=token,
        name="Mike",
        system_prompt=SYSTEM_PROMPT,
        voice_id=VOICE_ID,
        goes_first=False,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--room-url", required=True)
    p.add_argument("--token", required=True)
    args = p.parse_args()
    try:
        asyncio.run(main(args.room_url, args.token))
    except KeyboardInterrupt:
        logger.info("[Mike] stopped.")
