"""Latency metrics observer for the two-bots pipeline.

Collects per-turn TTFB, processing time, user->bot response latency,
and per-turn transcription (what was heard + what the bot replied).

Broadcasts events via Daily app-messages to the browser. Appends to
metrics_store for bot.py /metrics. Logs aggregate summary at session end.
"""

import json
import os
import threading
import time
import uuid
import urllib.request
from collections import defaultdict
from statistics import mean, median
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    TextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    MetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection

# Optional: POST per-turn records to e.g. https://yourapp.com/api/metrics for SessionMetric DB.
_METRICS_INGEST_URL = os.getenv("METRICS_INGEST_URL", "")
_DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")


def _post(url: str, data: dict, *, headers: dict | None = None, retries: int = 0):
    """POST with optional retries for critical endpoints (Daily app-messages)."""
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    is_daily = "api.daily.co" in url and "/send-app-message" in url
    max_attempts = (retries + 1) if is_daily else 1

    def _send():
        payload = json.dumps(data).encode()
        for attempt in range(max_attempts):
            try:
                req = urllib.request.Request(url, data=payload, headers=hdrs)
                urllib.request.urlopen(req, timeout=3)
                return
            except Exception as e:
                if is_daily:
                    body = ""
                    if hasattr(e, "read"):
                        try:
                            body = e.read().decode()[:200]
                        except Exception:
                            pass
                    if attempt < max_attempts - 1:
                        time.sleep(0.3 * (attempt + 1))
                    else:
                        logger.warning(
                            f"Daily send-app-message failed after {max_attempts} "
                            f"attempts: {e} | {body}"
                        )

    threading.Thread(target=_send, daemon=True).start()


def _p95(values: list[float]) -> float:
    """Return the 95th-percentile value (nearest-rank)."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * 0.95), len(s) - 1)
    return s[idx]


def _fmt(seconds: float) -> str:
    """Format a duration: ms when < 1 s, otherwise seconds."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.3f}s"


def _stats_line(label: str, values: list[float]) -> str:
    """One-line summary: avg / p50 / p95 / min / max (n=...)."""
    return (
        f"  {label:.<30s} avg={_fmt(mean(values))}  "
        f"p50={_fmt(median(values))}  p95={_fmt(_p95(values))}  "
        f"min={_fmt(min(values))}  max={_fmt(max(values))}  (n={len(values)})"
    )


def _stats_dict(values: list[float]) -> dict:
    """Return a JSON-friendly stats summary."""
    return {
        "avg": mean(values),
        "p50": median(values),
        "p95": _p95(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _is_tts_processor(processor: str) -> bool:
    """Heuristic: does this processor name belong to a TTS service?"""
    low = processor.lower()
    return any(tok in low for tok in ("tts", "elevenlabs", "cartesia", "xtts", "playht"))


class LatencyMetricsObserver(BaseObserver):
    """Aggregates TTFB, processing, user->bot latency, and per-turn
    transcription for each agent.

    Attach as an observer on the PipelineTask.  At session end it logs a
    table of percentile statistics for every metric it saw.

    When *metrics_store* is provided, a consolidated per-utterance record
    is also appended there (used by the PCC Session API ``/metrics``
    endpoint in ``bot.py`` and by the local dev server).
    """

    def __init__(
        self,
        agent_name: str = "",
        *,
        session_id: str = "",
        metrics_store: list[dict[str, Any]] | None = None,
        metrics_url: str | None = None,
        room_name: str = "",
    ):
        super().__init__()
        self._agent_name = agent_name
        self._session_id = session_id or f"{agent_name}-{int(time.time())}"
        self._metrics_store = metrics_store
        self._metrics_ingest_url = metrics_url or _METRICS_INGEST_URL
        self._room_name = room_name
        self._seen_frames: set[str] = set()
        self._event_history: list[dict] = []
        logger.info(
            f"[{agent_name}] LatencyMetricsObserver init: "
            f"room_name={room_name!r}, daily_key={'set' if _DAILY_API_KEY else 'MISSING'}"
        )

        # Per-processor series:  {"OpenAI LLM": [0.32, 0.28, ...], ...}
        self._ttfb: dict[str, list[float]] = defaultdict(list)
        self._processing: dict[str, list[float]] = defaultdict(list)

        # User -> bot end-to-end latency
        self._user_stopped_at: float = 0.0
        self._e2e_latencies: list[float] = []

        # Per-turn transcription tracking
        self._turn_count: int = 0
        self._current_input_chunks: list[str] = []
        self._current_output_chunks: list[str] = []
        self._collecting_response: bool = False
        self._llm_processor: object | None = None
        self._last_e2e: float | None = None

        # Buffered per-turn pipeline metrics for consolidated record
        self._pending_ttfb: float | None = None
        self._pending_llm: float | None = None
        self._pending_tts: float | None = None

        # Interruption tracking
        self._bot_speaking: bool = False
        self._interruption_count: int = 0

    def _broadcast(self, data: dict):
        """Broadcast metrics via Daily app-message to all room participants.

        The Daily REST API ``POST /rooms/:name/send-app-message`` delivers
        the payload to the browser (CallProvider) for live display. Metrics
        are persisted when the transcript is saved.
        """
        payload = {**data, "label": "metrics", "id": str(uuid.uuid4())}
        self._event_history.append(data)
        if self._room_name and _DAILY_API_KEY:
            url = f"https://api.daily.co/v1/rooms/{self._room_name}/send-app-message"
            _post(
                url,
                {"data": payload, "recipient": "*"},
                headers={"Authorization": f"Bearer {_DAILY_API_KEY}"},
                retries=2,
            )

    def broadcast_history(self):
        """Re-broadcast all past events so a late-joining browser can catch up."""
        if not self._room_name or not _DAILY_API_KEY:
            return
        url = f"https://api.daily.co/v1/rooms/{self._room_name}/send-app-message"
        for data in self._event_history:
            _post(
                url,
                {"data": {**data, "label": "metrics", "id": str(uuid.uuid4()), "replay": True}, "recipient": "*"},
                headers={"Authorization": f"Bearer {_DAILY_API_KEY}"},
                retries=1,
            )

    # ------------------------------------------------------------------
    # Observer hook
    # ------------------------------------------------------------------

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame

        # --- deduplicate (same frame seen at multiple processor hops) ---
        fid = frame.id
        if fid in self._seen_frames:
            return
        self._seen_frames.add(fid)

        # --- metrics frames (TTFB / processing) ---
        if isinstance(frame, MetricsFrame):
            for md in frame.data:
                self._record_metrics(md)
            return

        # --- only look at downstream frames from here ---
        if data.direction != FrameDirection.DOWNSTREAM:
            return

        # --- transcription (what the other agent said) ---
        if isinstance(frame, TranscriptionFrame):
            self._current_input_chunks.append(frame.text)
            return

        # --- LLM response bracketing ---
        if isinstance(frame, LLMFullResponseStartFrame):
            self._collecting_response = True
            self._llm_processor = data.source
            self._current_output_chunks = []
            return

        if isinstance(frame, TextFrame) and self._collecting_response:
            if data.source is self._llm_processor:
                self._current_output_chunks.append(frame.text)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            self._collecting_response = False
            self._llm_processor = None
            self._emit_turn()
            return

        # --- bot speaking state ---
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            if self._user_stopped_at:
                latency = time.time() - self._user_stopped_at
                self._user_stopped_at = 0.0
                self._e2e_latencies.append(latency)
                self._last_e2e = latency
                tag = f"[{self._agent_name}] " if self._agent_name else ""
                logger.info(f"{tag}user->bot latency: {_fmt(latency)}")
                self._broadcast({
                    "type": "e2e",
                    "agent": self._agent_name,
                    "value": latency,
                    "ts": time.time(),
                })
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            return

        # --- user -> bot latency + interruption detection ---
        if isinstance(frame, VADUserStartedSpeakingFrame):
            if self._bot_speaking:
                self._interruption_count += 1
                tag = f"[{self._agent_name}] " if self._agent_name else ""
                logger.info(
                    f"{tag}INTERRUPTION #{self._interruption_count} — "
                    f"user started speaking while bot was talking"
                )
                self._broadcast({
                    "type": "interruption",
                    "agent": self._agent_name,
                    "count": self._interruption_count,
                    "ts": time.time(),
                })
            self._user_stopped_at = 0.0
            return

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_stopped_at = time.time()
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            self._log_summary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit_turn(self):
        """Emit a turn event with input/output transcription and flush
        buffered pipeline metrics into a consolidated record."""
        input_text = " ".join(self._current_input_chunks).strip()
        output_text = "".join(self._current_output_chunks).strip()

        if not input_text and not output_text:
            self._current_input_chunks = []
            self._current_output_chunks = []
            return

        self._turn_count += 1

        tag = f"[{self._agent_name}] " if self._agent_name else ""
        logger.info(
            f"{tag}turn {self._turn_count}: "
            f"heard={input_text[:80]!r} -> replied={output_text[:80]!r}"
        )

        self._broadcast({
            "type": "turn",
            "agent": self._agent_name,
            "turn": self._turn_count,
            "input": input_text,
            "output": output_text,
            "e2e": self._last_e2e,
            "ts": time.time(),
        })

        # Consolidated record for metrics store / webhook
        def _ms(v: float | None) -> float | None:
            return round(v * 1000, 2) if v is not None else None

        record: dict[str, Any] = {
            "session_id": self._session_id,
            "agent_name": self._agent_name,
            "turn_index": self._turn_count - 1,
            "input": input_text,
            "output": output_text,
            "ttfb": _ms(self._pending_ttfb),
            "llm_duration": _ms(self._pending_llm),
            "tts_duration": _ms(self._pending_tts),
            "user_bot_latency": _ms(self._last_e2e),
            "e2e_latency": _ms(self._last_e2e),
            "ts": time.time(),
        }

        if self._metrics_store is not None:
            self._metrics_store.append(record)
        if self._metrics_ingest_url:
            _post(self._metrics_ingest_url, record)

        # Reset per-turn state
        self._current_input_chunks = []
        self._current_output_chunks = []
        self._last_e2e = None
        self._pending_ttfb = None
        self._pending_llm = None
        self._pending_tts = None

    def _record_metrics(self, md: MetricsData):
        label = md.processor
        if md.model:
            label += f" ({md.model})"

        is_tts = _is_tts_processor(md.processor)

        if isinstance(md, TTFBMetricsData):
            self._ttfb[label].append(md.value)
            if not is_tts:
                self._pending_ttfb = md.value
            self._broadcast({
                "type": "ttfb",
                "agent": self._agent_name,
                "processor": label,
                "value": md.value,
                "ts": time.time(),
            })
        elif isinstance(md, ProcessingMetricsData):
            self._processing[label].append(md.value)
            if is_tts:
                self._pending_tts = md.value
            else:
                self._pending_llm = md.value
            self._broadcast({
                "type": "processing",
                "agent": self._agent_name,
                "processor": label,
                "value": md.value,
                "ts": time.time(),
            })

    def _log_summary(self):
        tag = f"[{self._agent_name}] " if self._agent_name else ""
        header = f"{tag}Latency summary"
        sections: list[str] = []

        if self._ttfb:
            lines = [_stats_line(k, v) for k, v in sorted(self._ttfb.items())]
            sections.append("TTFB\n" + "\n".join(lines))

        if self._processing:
            lines = [_stats_line(k, v) for k, v in sorted(self._processing.items())]
            sections.append("Processing\n" + "\n".join(lines))

        if self._e2e_latencies:
            sections.append(
                "User -> Bot (end-to-end)\n"
                + _stats_line("e2e", self._e2e_latencies)
            )

        if not sections:
            logger.info(f"{header}: no latency data collected")
            return

        body = "\n".join(sections)
        logger.info(f"{header}\n{body}")

        summary: dict = {}
        if self._ttfb:
            summary["ttfb"] = {k: _stats_dict(v) for k, v in self._ttfb.items()}
        if self._processing:
            summary["processing"] = {k: _stats_dict(v) for k, v in self._processing.items()}
        if self._e2e_latencies:
            summary["e2e"] = _stats_dict(self._e2e_latencies)
        if summary:
            self._broadcast({
                "type": "summary",
                "agent": self._agent_name,
                "data": summary,
                "ts": time.time(),
            })
