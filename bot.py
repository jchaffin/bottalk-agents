# Copyright 2026 Jacob Chaffin / Outrival. All rights reserved.

"""
Pipecat Cloud entry point.

Receives session arguments from the platform and launches a voice
agent into a shared Daily room.

Since we create the Daily room ourselves (to put both agents in
the same room), we do NOT use ``createDailyRoom`` in the start
request.  All config comes via ``args.body``.

A ``/metrics`` endpoint is registered via the PCC Session API so
callers can fetch live latency data for this session without a DB
round-trip::

    GET /v1/public/{agent}/sessions/{sessionId}/metrics
"""

from typing import Any

from loguru import logger
from pipecat.runner.types import RunnerArguments

from agent import run_agent
from sarah import CONFIG as SARAH_CONFIG, get_voice_config as sarah_voice_config
from mike import CONFIG as MIKE_CONFIG, get_voice_config as mike_voice_config

AGENT_CONFIGS = {
    "Sarah": SARAH_CONFIG,
    "Mike": MIKE_CONFIG,
}

VOICE_CONFIGS = {
    "Sarah": sarah_voice_config,
    "Mike": mike_voice_config,
}

# ---------------------------------------------------------------------------
# Per-session metrics store
# ---------------------------------------------------------------------------

_session_metrics: list[dict[str, Any]] = []
_session_kpis: dict[str, Any] = {}
_session_info: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# PCC Session API endpoints (only available on Pipecat Cloud)
# ---------------------------------------------------------------------------

try:
    from pipecatcloud_system import app  # type: ignore[import-untyped]

    @app.get("/metrics")
    async def get_metrics():
        """Return accumulated latency metrics for this session."""
        items = list(_session_metrics)
        count = len(items)

        def _vals(key: str) -> list[float]:
            return [m[key] for m in items if m.get(key) is not None]

        def _avg(vals: list[float]) -> float | None:
            return round(sum(vals) / len(vals), 2) if vals else None

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
            "session": _session_info,
            "count": count,
            "aggregates": {
                "ttfb": {"avg": _avg(ttfb), "p95": _p95(ttfb)},
                "llm": {"avg": _avg(llm), "p95": _p95(llm)},
                "tts": {"avg": _avg(tts), "p95": _p95(tts)},
                "e2e": {"avg": _avg(e2e), "p95": _p95(e2e)},
                "user_bot_latency": {"avg": _avg(ubl), "p95": _p95(ubl)},
            },
            "kpis": dict(_session_kpis),
            "timeseries": items,
        }

    @app.get("/metrics/summary")
    async def get_metrics_summary():
        """Lightweight summary — just aggregates, no raw timeseries."""
        items = list(_session_metrics)

        def _vals(key: str) -> list[float]:
            return [m[key] for m in items if m.get(key) is not None]

        def _avg(vals: list[float]) -> float | None:
            return round(sum(vals) / len(vals), 2) if vals else None

        ttfb = _vals("ttfb")
        llm = _vals("llm_duration")
        tts = _vals("tts_duration")
        e2e = _vals("e2e_latency")
        ubl = _vals("user_bot_latency")

        return {
            "session": _session_info,
            "count": len(items),
            "ttfb_avg": _avg(ttfb),
            "llm_avg": _avg(llm),
            "tts_avg": _avg(tts),
            "e2e_avg": _avg(e2e),
            "user_bot_latency_avg": _avg(ubl),
            "kpis": dict(_session_kpis),
        }

    @app.get("/kpis")
    async def get_kpis():
        """Return conversation-level KPIs for this session."""
        return {
            "session": _session_info,
            "kpis": dict(_session_kpis),
        }

    logger.info("PCC Session API: /metrics, /metrics/summary, and /kpis registered")

except ImportError:
    logger.debug("pipecatcloud_system not available (local dev mode)")


# ---------------------------------------------------------------------------
# Bot entry point
# ---------------------------------------------------------------------------


async def bot(args: RunnerArguments):
    body = args.body or {}

    room_url = body["room_url"]
    token = body["token"]
    name = body.get("name", "Sarah")
    defaults = AGENT_CONFIGS.get(name, {})

    system_prompt = body.get("system_prompt") or defaults.get("system_prompt", "")
    voice_id = body.get("voice_id") or defaults.get("voice_id", "")
    goes_first = body.get("goes_first", defaults.get("goes_first", False))
    known_agents = set(body.get("known_agents", []))
    max_turns = body.get("max_turns", 20)

    session_id = getattr(args, "session_id", "local")

    _session_info.update({
        "session_id": session_id,
        "agent_name": name,
        "room_url": room_url,
        "goes_first": goes_first,
    })

    # Agent-specific VAD / SmartTurn config
    voice_cfg_fn = VOICE_CONFIGS.get(name)
    voice_kwargs = voice_cfg_fn() if voice_cfg_fn else {}

    logger.info(
        f"[{name}] starting session {session_id} in {room_url} "
        f"(max_turns={max_turns}, voice_config={list(voice_kwargs.keys())})"
    )

    await run_agent(
        room_url=room_url,
        token=token,
        name=name,
        system_prompt=system_prompt,
        voice_id=voice_id,
        goes_first=goes_first,
        known_agents=known_agents or None,
        max_turns=max_turns,
        session_id=session_id,
        metrics_store=_session_metrics,
        kpi_store=_session_kpis,
        **voice_kwargs,
    )
