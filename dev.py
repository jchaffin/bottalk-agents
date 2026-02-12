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

TOPIC = os.getenv("CONVERSATION_TOPIC", "enterprise software sales")

SARAH_PROMPT = (
    f"You are Sarah, an enterprise software sales rep at TechFlow Solutions. "
    f"You are on a live phone call with a potential customer about {TOPIC}.\n\n"
    "Your product — TechFlow — is an AI workflow automation platform that "
    "integrates with Salesforce, HubSpot, Slack, Jira, and 50+ other tools. "
    "Professional tier: $99/user/month. 30-day free trial. "
    "Case study: Acme Corp cut manual work by 60% in 3 months.\n\n"
    "Rules:\n"
    "- 2-3 short spoken sentences per turn. No bullets, no markdown, no emoji.\n"
    "- Be warm, curious, empathetic. Ask questions. Handle objections gracefully.\n"
    "- Goal: understand pain, demo value, propose a free trial or a next call."
)

MIKE_PROMPT = (
    f"You are Mike, VP of Ops at BrightCart, a 200-person e-commerce company. "
    f"A sales rep just called you about {TOPIC}.\n\n"
    "Your pain: manual order processing, poor tool integration, team drowning "
    "in repetitive tasks. Budget ~$50k/yr. Last year you bought an expensive "
    "platform that flopped, so you are cautious.\n\n"
    "Rules:\n"
    "- 2-3 short spoken sentences per turn. No bullets, no markdown, no emoji.\n"
    "- Be interested but skeptical. Push back on price. Ask for proof of ROI.\n"
    "- Do not agree too quickly. Ask pointed questions about timeline and support."
)

SARAH_VOICE = os.getenv("SARAH_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
MIKE_VOICE = os.getenv("MIKE_VOICE_ID", "TxGEqnHWrfWFTfGW9XjX")

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
async def start():
    for key in ("DAILY_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY"):
        if not os.getenv(key):
            raise HTTPException(500, f"Missing {key}")

    _terminate()

    async with aiohttp.ClientSession() as session:
        helper = DailyRESTHelper(
            daily_api_key=os.getenv("DAILY_API_KEY"),
            aiohttp_session=session,
        )
        room = await helper.create_room(
            DailyRoomParams(properties=DailyRoomProperties(exp=time.time() + 600))
        )
        t_sarah = await helper.get_token(room.url)
        t_mike = await helper.get_token(room.url)
        t_browser = await helper.get_token(room.url)

    room_url = room.url
    py = sys.executable
    here = os.path.dirname(__file__)
    logger.info(f"Room: {room_url}")

    # Spawn as fully separate processes (not multiprocessing) so each
    # gets its own Daily client with no shared state.
    sarah = subprocess.Popen([
        py, os.path.join(here, "run_agent.py"),
        "--room-url", room_url,
        "--token", t_sarah,
        "--name", "Sarah",
        "--system-prompt", SARAH_PROMPT,
        "--voice-id", SARAH_VOICE,
        "--goes-first",
    ])
    _procs.append(sarah)

    await asyncio.sleep(3)

    mike = subprocess.Popen([
        py, os.path.join(here, "run_agent.py"),
        "--room-url", room_url,
        "--token", t_mike,
        "--name", "Mike",
        "--system-prompt", MIKE_PROMPT,
        "--voice-id", MIKE_VOICE,
    ])
    _procs.append(mike)

    logger.info(f"Sarah PID={sarah.pid}  Mike PID={mike.pid}")
    return {"roomUrl": room_url, "token": t_browser}


@app.post("/api/stop")
async def stop():
    _terminate()
    return {"status": "stopped"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
