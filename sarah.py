"""
Sarah — sales rep participant.

    python sarah.py --room-url https://… --token …
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADParams

load_dotenv(".env.local", override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG", filter=lambda r: "rtvi" not in r["name"] and "vad" not in r["name"])

TOPIC = os.getenv("CONVERSATION_TOPIC", "enterprise software sales")

SYSTEM_PROMPT = f"""\
You are Sarah, an enterprise software sales rep at TechFlow Solutions. \
You are on a live phone call with a potential customer about {TOPIC}.

Your product — TechFlow — is an AI workflow automation platform that \
integrates with Salesforce, HubSpot, Slack, Jira, and 50+ other tools. \
Professional tier: $99/user/month. 30-day free trial. \
Case study: Acme Corp cut manual work by 60% in 3 months.

Rules:
- 2-3 short spoken sentences per turn. No bullets, no markdown, no emoji.
- Be warm, curious, empathetic. Ask questions. Handle objections gracefully.
- Goal: understand pain, demo value, propose a free trial or a next call."""

VOICE_ID = os.getenv("SARAH_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

CONFIG = {
    "system_prompt": SYSTEM_PROMPT,
    "voice_id": VOICE_ID,
    "goes_first": True,
}


async def main(room_url: str, token: str):
    from agent import run_agent

    logger.info(f"[Sarah] joining {room_url}")
    await run_agent(
        room_url=room_url,
        token=token,
        name="Sarah",
        system_prompt=SYSTEM_PROMPT,
        voice_id=VOICE_ID,
        goes_first=True,
        known_agents={"Sarah", "Mike"},
        max_turns=20,
        allow_interruptions=True,
        vad_params=VADParams(
            threshold=0.6,
            min_volume=0.4,
            stop_secs=1.5,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--room-url", required=True)
    p.add_argument("--token", required=True)
    args = p.parse_args()
    try:
        asyncio.run(main(args.room_url, args.token))
    except KeyboardInterrupt:
        logger.info("[Sarah] stopped.")
