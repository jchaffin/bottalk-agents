"""
Pipecat Cloud entry point.

Receives session arguments from the platform and launches a voice
agent into a shared Daily room.

Since we create the Daily room ourselves (to put both agents in
the same room), we do NOT use ``createDailyRoom`` in the start
request.  All config comes via ``args.body``.
"""

import asyncio

from loguru import logger
from pipecat.runner.types import RunnerArguments

from agent import run_agent

# Seconds the second agent waits before joining. This gives the first
# agent time to join the Daily room and start room-level transcription.
_SECOND_AGENT_DELAY = 5


async def bot(args: RunnerArguments):
    body = args.body or {}

    room_url = body["room_url"]
    token = body["token"]
    name = body["name"]
    system_prompt = body["system_prompt"]
    voice_id = body["voice_id"]
    goes_first = body.get("goes_first", False)
    known_agents = set(body.get("known_agents", []))

    session_id = getattr(args, "session_id", "local")
    logger.info(f"[{name}] starting session {session_id} in {room_url}")

    # Let the first agent join and start transcription before the second
    # agent enters the room (mirrors the 3-second sleep in dev.py).
    if not goes_first:
        logger.info(
            f"[{name}] waiting {_SECOND_AGENT_DELAY}s for first agent "
            "to start transcription"
        )
        await asyncio.sleep(_SECOND_AGENT_DELAY)

    await run_agent(
        room_url=room_url,
        token=token,
        name=name,
        system_prompt=system_prompt,
        voice_id=voice_id,
        goes_first=goes_first,
        known_agents=known_agents or None,
    )
