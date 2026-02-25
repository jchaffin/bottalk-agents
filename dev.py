# Copyright 2026 Jacob Chaffin / bottalk. All rights reserved.

"""
Local dev server — runs agents as separate processes, no Pipecat Cloud needed.

    cd agents && pip install -r requirements.txt
    python dev.py

Frontend connects via NEXT_PUBLIC_API_URL=http://localhost:8000
"""

import asyncio
import importlib
import os
import subprocess
import sys
import time
from typing import Any

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"), override=True)
load_dotenv(".env.local", override=True)

daily_utils = importlib.import_module("pipecat.transports.daily.utils")
DailyRESTHelper = daily_utils.DailyRESTHelper
DailyRoomParams = daily_utils.DailyRoomParams
DailyRoomProperties = daily_utils.DailyRoomProperties

logger.remove(0)
logger.add(sys.stderr, level="DEBUG", filter=lambda r: "vad" not in r["name"])

DEFAULT_VOICE_1 = os.getenv("DEFAULT_VOICE_1", "21m00Tcm4TlvDq8ikWAM")
DEFAULT_VOICE_2 = os.getenv("DEFAULT_VOICE_2", "TxGEqnHWrfWFTfGW9XjX")

app = FastAPI(title="bottalk Local Dev")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_procs: list[subprocess.Popen] = []
_metrics: list[dict[str, Any]] = []

# WebSocket clients for live transcript relay (mirrors two-bots/server.py)
_ws_clients: set[WebSocket] = set()
_transcript_events: list[dict[str, Any]] = []


async def _broadcast_ws(data: dict):
    """Send an event to all connected WebSocket clients."""
    dead: list[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


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
async def start(request: Request):
    for key in ("DAILY_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY"):
        if not os.getenv(key):
            raise HTTPException(500, f"Missing {key}")

    _terminate()
    _transcript_events.clear()

    try:
        body = await request.json()
    except Exception:
        body = {}

    # Agents can be provided by the frontend as an array of two objects.
    # Each agent has name, prompt, voice_id, role. Prompts are sent through to the agent backend.
    agents = body.get("agents")

    if not agents or len(agents) < 2:
        agents = [
            {"name": "System", "prompt": "", "voice_id": DEFAULT_VOICE_1},
            {"name": "User", "prompt": "", "voice_id": DEFAULT_VOICE_2},
        ]

    agent1 = dict(agents[0]) if isinstance(agents[0], dict) else {}
    agent2 = dict(agents[1]) if isinstance(agents[1], dict) else {}

    agent1.setdefault("name", "System")
    agent2.setdefault("name", "User")
    agent1.setdefault("prompt", "")
    agent2.setdefault("prompt", "")
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
        t1, t2, t_browser = await asyncio.gather(
            helper.get_token(room.url),
            helper.get_token(room.url),
            helper.get_token(room.url),
        )

    room_url = room.url
    py = sys.executable
    here = os.path.dirname(__file__)
    logger.info(f"Room: {room_url}")

    # Spawn bot.py twice with full agent config — prompts come from frontend, not hardcoded.
    # bot.py falls back to config.AGENT_CONFIGS when prompt is empty (Quick Start).
    proc1 = subprocess.Popen([
        py, os.path.join(here, "bot.py"),
        "--room-url", room_url,
        "--token", t1,
        "--name", agent1["name"],
        "--system-prompt", agent1["prompt"],
        "--voice-id", voice1,
        "--goes-first",
        "--known-agents", all_names,
        "--session-id", "local-1",
    ])
    _procs.append(proc1)

    proc2 = subprocess.Popen([
        py, os.path.join(here, "bot.py"),
        "--room-url", room_url,
        "--token", t2,
        "--name", agent2["name"],
        "--system-prompt", agent2["prompt"],
        "--voice-id", voice2,
        "--known-agents", all_names,
        "--session-id", "local-2",
    ])
    _procs.append(proc2)

    logger.info(f"{agent1['name']} PID={proc1.pid}  {agent2['name']} PID={proc2.pid}")
    return {"roomUrl": room_url, "token": t_browser}


@app.post("/api/stop")
async def stop():
    _terminate()
    return {"status": "stopped"}


# ---------------------------------------------------------------------------
# WebSocket transcript relay (mirrors two-bots/server.py)
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def ws_transcript(ws: WebSocket):
    """WebSocket endpoint for live transcript events."""
    await ws.accept()
    _ws_clients.add(ws)

    # Replay event history so late-joining browsers see past turns
    for event in _transcript_events:
        try:
            await ws.send_json(event)
        except Exception:
            break

    try:
        async for _ in ws.iter_text():
            pass  # no inbound messages expected
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


# ---------------------------------------------------------------------------
# Metrics endpoints (local dev — mirrors the Next.js /api/metrics route)
# ---------------------------------------------------------------------------


@app.post("/api/metrics")
async def ingest_metrics(request: Request):
    """Receive latency/transcript events from agents.

    Accepts two formats:
      1. Dashboard-format events from LatencyMetricsObserver (type: turn, e2e, etc.)
         — stored in _transcript_events and broadcast via WebSocket.
      2. Consolidated per-turn records (session_id / agent_name / ttfb / ...)
         — stored in _metrics for the /api/metrics GET endpoint.
    """
    body = await request.json()

    # Dashboard-format events (turn, e2e, ttfb, processing, summary)
    if "type" in body:
        _transcript_events.append(body)
        await _broadcast_ws(body)
        return {"ok": True}

    # Consolidated per-turn metrics format
    session_id = body.get("session_id")
    agent_name = body.get("agent_name")
    if not session_id or not agent_name:
        raise HTTPException(400, "session_id and agent_name are required")

    body["received_at"] = time.time()
    _metrics.append(body)
    await _broadcast_ws(body)
    logger.info(f"[metrics] ingested turn {body.get('turn_index', '?')} from {agent_name}")
    return {"status": "ok", "count": len(_metrics)}


@app.get("/api/metrics")
async def get_metrics(session_id: str | None = None):
    """Return aggregated latency metrics from the in-memory store."""
    items = [m for m in _metrics if not session_id or m.get("session_id") == session_id]
    count = len(items)

    def _vals(key: str) -> list[float]:
        return [m[key] for m in items if m.get(key) is not None]

    def _avg(vals: list[float]) -> float | None:
        return round(sum(vals) / len(vals), 2) if vals else None

    def _p50(vals: list[float]) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        return round(s[len(s) // 2], 2)

    def _p95(vals: list[float]) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        return round(s[int(len(s) * 0.95)], 2)

    ttfb = _vals("ttfb")
    llm = _vals("llm_duration")
    tts = _vals("tts_duration")
    e2e = _vals("e2e_latency")
    ubl = _vals("user_bot_latency")

    return {
        "count": count,
        "aggregates": {
            "ttfb": {"avg": _avg(ttfb), "p50": _p50(ttfb), "p95": _p95(ttfb)},
            "llm": {"avg": _avg(llm), "p50": _p50(llm), "p95": _p95(llm)},
            "tts": {"avg": _avg(tts), "p50": _p50(tts), "p95": _p95(tts)},
            "e2e": {"avg": _avg(e2e), "p50": _p50(e2e), "p95": _p95(e2e)},
            "user_bot_latency": {"avg": _avg(ubl), "p50": _p50(ubl), "p95": _p95(ubl)},
        },
        "timeseries": items,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
