"""
Pipecat Cloud entry point — custom image without RTVI processor.

The default PCC base image wraps agents in an RTVI framework that
validates all incoming Daily app-messages.  When two agents share
a room, each agent's messages flood the other's RTVI processor
with validation errors, choking the event loop.

This custom image is a plain FastAPI server that exposes POST /bot
(the endpoint PCC calls to start a session) and runs the agent
directly — no RTVI layer.
"""

import asyncio
import os

from fastapi import FastAPI, Request
from loguru import logger
import uvicorn

from agent import run_agent

app = FastAPI()


@app.post("/bot")
async def start_bot(request: Request):
    body = await request.json()
    data = body.get("body", body)

    room_url = data.get("room_url")
    token = data.get("token")
    name = data["name"]
    system_prompt = data["system_prompt"]
    voice_id = data["voice_id"]
    goes_first = data.get("goes_first", False)

    logger.info(f"[{name}] starting in {room_url}")

    # Run the agent in a background task so the HTTP response returns immediately
    asyncio.create_task(
        run_agent(
            room_url=room_url,
            token=token,
            name=name,
            system_prompt=system_prompt,
            voice_id=voice_id,
            goes_first=goes_first,
        )
    )

    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
