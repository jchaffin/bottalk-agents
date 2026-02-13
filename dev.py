"""
Local dev server — runs agents as separate processes, no Pipecat Cloud needed.

    cd agents && pip install -r requirements.txt
    python dev.py

Frontend connects via NEXT_PUBLIC_API_URL=http://localhost:8000
"""

import asyncio
import os
import subprocess
import sys
import time

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"), override=True)
load_dotenv(".env.local", override=True)

from pipecat.transports.daily.utils import (
    DailyRESTHelper,
    DailyRoomParams,
    DailyRoomProperties,
)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

DEFAULT_VOICE_1 = os.getenv("DEFAULT_VOICE_1", "21m00Tcm4TlvDq8ikWAM")
DEFAULT_VOICE_2 = os.getenv("DEFAULT_VOICE_2", "TxGEqnHWrfWFTfGW9XjX")

app = FastAPI(title="Outrival Local Dev")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_procs: list[subprocess.Popen] = []


def _terminate():
    for p in _procs:
        if p.poll() is None:
            p.terminate()
    for p in _procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    _procs.clear()


@app.post("/api/start")
async def start(request: dict | None = None):
    for key in ("DAILY_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY"):
        if not os.getenv(key):
            raise HTTPException(500, f"Missing {key}")

    _terminate()

    # Agents must be provided by the frontend as an array of two objects,
    # each with at least { name, prompt }.  voice_id is optional.
    body = request or {}
    agents = body.get("agents")
    if not agents or len(agents) < 2:
        raise HTTPException(400, "agents array with two entries is required")

    agent1 = agents[0]
    agent2 = agents[1]

    if not agent1.get("name") or not agent1.get("prompt"):
        raise HTTPException(400, "agents[0] must have name and prompt")
    if not agent2.get("name") or not agent2.get("prompt"):
        raise HTTPException(400, "agents[1] must have name and prompt")

    voice1 = agent1.get("voice_id") or DEFAULT_VOICE_1
    voice2 = agent2.get("voice_id") or DEFAULT_VOICE_2
    all_names = f"{agent1['name']},{agent2['name']}"

    async with aiohttp.ClientSession() as session:
        helper = DailyRESTHelper(
            daily_api_key=os.getenv("DAILY_API_KEY"),
            aiohttp_session=session,
        )
        room = await helper.create_room(
            DailyRoomParams(properties=DailyRoomProperties(exp=time.time() + 600))
        )
        t1 = await helper.get_token(room.url)
        t2 = await helper.get_token(room.url)
        t_browser = await helper.get_token(room.url)

    room_url = room.url
    py = sys.executable
    here = os.path.dirname(__file__)
    logger.info(f"Room: {room_url}")

    # Spawn as fully separate processes (not multiprocessing) so each
    # gets its own Daily client with no shared state.
    proc1 = subprocess.Popen([
        py, os.path.join(here, "run_agent.py"),
        "--room-url", room_url,
        "--token", t1,
        "--name", agent1["name"],
        "--system-prompt", agent1["prompt"],
        "--voice-id", voice1,
        "--goes-first",
        "--known-agents", all_names,
    ])
    _procs.append(proc1)

    await asyncio.sleep(3)

    proc2 = subprocess.Popen([
        py, os.path.join(here, "run_agent.py"),
        "--room-url", room_url,
        "--token", t2,
        "--name", agent2["name"],
        "--system-prompt", agent2["prompt"],
        "--voice-id", voice2,
        "--known-agents", all_names,
    ])
    _procs.append(proc2)

    logger.info(f"{agent1['name']} PID={proc1.pid}  {agent2['name']} PID={proc2.pid}")
    return {"roomUrl": room_url, "token": t_browser}


@app.post("/api/stop")
async def stop():
    _terminate()
    return {"status": "stopped"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
