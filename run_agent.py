# Copyright 2026 Jacob Chaffin / Outrival. All rights reserved.

"""
CLI entry point for running a single agent locally.

    python run_agent.py --room-url URL --token TOKEN --name Sarah \
        --system-prompt "..." --voice-id "..." --goes-first \
        --known-agents "Sarah,Mike"
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"), override=True)
load_dotenv(".env.local", override=True)

from loguru import logger

logger.remove(0)
logger.add(sys.stderr, level="DEBUG", filter=lambda r: "rtvi" not in r["name"])


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--room-url", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--system-prompt", required=True)
    p.add_argument("--voice-id", required=True)
    p.add_argument("--goes-first", action="store_true")
    p.add_argument(
        "--known-agents",
        default="",
        help="Comma-separated list of all agent names in the room",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Max LLM response turns before the agent shuts down (default: 20)",
    )
    args = p.parse_args()

    known = (
        set(n.strip() for n in args.known_agents.split(",") if n.strip())
        if args.known_agents
        else None
    )

    from agent import run_agent

    await run_agent(
        room_url=args.room_url,
        token=args.token,
        name=args.name,
        system_prompt=args.system_prompt,
        voice_id=args.voice_id,
        goes_first=args.goes_first,
        known_agents=known,
        max_turns=args.max_turns,
    )


if __name__ == "__main__":
    asyncio.run(main())
