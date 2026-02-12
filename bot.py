"""
Pipecat Cloud entry point.

Receives session arguments from the platform and launches the
appropriate voice agent (Sarah or Mike) into a shared Daily room.

Since we create the Daily room ourselves (to put both agents in
the same room), we do NOT use ``createDailyRoom`` in the start
request.  PCC therefore sends ``PipecatRunnerArguments`` — all
config comes via ``args.body``.

    {
        "room_url": "https://...",
        "token": "...",
        "name": "Sarah",
        "system_prompt": "...",
        "voice_id": "21m00Tcm4TlvDq8ikWAM",
        "goes_first": true
    }
"""

from loguru import logger
from pipecat.runner.types import PipecatRunnerArguments

from agent import run_agent


async def bot(args: PipecatRunnerArguments):
    body = args.body or {}

    room_url = body["room_url"]
    token = body["token"]
    name = body["name"]
    system_prompt = body["system_prompt"]
    voice_id = body["voice_id"]
    goes_first = body.get("goes_first", False)

    logger.info(f"[{name}] starting session {args.session_id} in {room_url}")

    await run_agent(
        room_url=room_url,
        token=token,
        name=name,
        system_prompt=system_prompt,
        voice_id=voice_id,
        goes_first=goes_first,
    )
