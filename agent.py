# Copyright 2026 Jacob Chaffin / Outrival. All rights reserved.

"""
Reusable Pipecat voice agent that joins a Daily room.

Multi-bot room handling
-----------------------
Daily's room-level Deepgram transcription broadcasts every participant's
speech to ALL participants.  Without filtering, each bot "hears" its own
TTS output echoed back as user input — infinite feedback loop.

We patch the transport's transcription callback to silently drop any
transcription whose ``participantId`` matches this bot.  When
``name_filter=True``, we additionally require the bot's name to appear
as a whole word in the transcription text.
"""

import asyncio
import os
import re
from typing import Any

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIObserverParams
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.daily.transport import DailyParams, DailyTransport

# Must match APP_MESSAGE_LABEL in frontend/src/lib/config.ts
APP_MESSAGE_LABEL = "outrival"


# ---------------------------------------------------------------------------
# Pipeline processors
# ---------------------------------------------------------------------------

class ConversationLimiter(FrameProcessor):
    """Hard-stop after *max_turns* LLM responses to prevent runaway usage."""

    def __init__(self, *, max_turns: int = 20):
        super().__init__()
        self._task: PipelineTask | None = None
        self._max_turns = max_turns
        self._turn_count = 0

    def set_task(self, task: PipelineTask):
        self._task = task

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseEndFrame):
            self._turn_count += 1
            logger.info(f"[Limiter] turn {self._turn_count}/{self._max_turns}")
            if self._turn_count >= self._max_turns and self._task:
                logger.warning(f"[Limiter] reached {self._max_turns} turns — stopping")
                await self._task.cancel()
                return
        await self.push_frame(frame, direction)


class ContextTrimmer(FrameProcessor):
    """Sliding-window trim on the LLM message list.

    Keeps the system prompt(s) and the last *max_turns* exchanges.
    Without trimming, TTFT grows linearly and the transcript visibly
    lags behind TTS audio after ~10 turns.
    """

    def __init__(self, messages: list, *, max_turns: int = 10):
        super().__init__()
        self._messages = messages
        self._max_turns = max_turns

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseEndFrame):
            self._trim()
        await self.push_frame(frame, direction)

    def _trim(self):
        limit = self._max_turns * 2
        n_sys = sum(1 for m in self._messages if m.get("role") == "system")
        n_convo = len(self._messages) - n_sys
        if n_convo > limit:
            drop = n_convo - limit
            del self._messages[n_sys : n_sys + drop]
            logger.debug(f"Trimmed {drop} msgs, kept {n_sys} sys + {n_convo - drop} convo")


class LLMTextRelay(FrameProcessor):
    """Batch LLM tokens into chunked ``bot-llm-text`` app-messages.

    Flushing every ~100 chars keeps the Daily signaling channel at
    2-3 msgs/sec instead of 50, preventing transcript lag.
    """

    _FLUSH_CHARS = 100

    def __init__(self):
        super().__init__()
        self._buffer = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = ""
            await self._send({"label": APP_MESSAGE_LABEL, "type": "bot-llm-started"})
        elif isinstance(frame, LLMTextFrame):
            self._buffer += frame.text
            if len(self._buffer) >= self._FLUSH_CHARS:
                await self._flush()
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._flush()
            await self._send({"label": APP_MESSAGE_LABEL, "type": "bot-llm-stopped"})

        await self.push_frame(frame, direction)

    async def _flush(self):
        if self._buffer:
            text, self._buffer = self._buffer, ""
            await self._send({
                "label": APP_MESSAGE_LABEL,
                "type": "bot-llm-text",
                "data": {"text": text},
            })

    async def _send(self, message: dict):
        await self.push_frame(OutputTransportMessageUrgentFrame(message=message))


# ---------------------------------------------------------------------------
# Transport patches
# ---------------------------------------------------------------------------

def _name_in_text(name: str, text: str) -> bool:
    """Check if *name* appears as a whole word (case-insensitive)."""
    return bool(re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE))


def _patch_app_message_filter(transport: DailyTransport):
    """Drop incoming app-messages with our label before RTVI validation.

    Without this, our custom messages fail RTVI validation and flood
    the event loop with errors, choking the pipeline.
    """
    orig = transport._client.on_app_message

    def filtered(message, sender):
        if isinstance(message, dict) and message.get("label") == APP_MESSAGE_LABEL:
            return
        orig(message, sender)

    transport._client.on_app_message = filtered


def _patch_transcription_filter(
    transport: DailyTransport, name: str, name_filter: bool,
) -> list[str]:
    """Filter self-transcription (and optionally non-addressed speech).

    Returns a mutable list for storing this bot's participant ID.
    """
    self_id: list[str] = []
    orig = transport._client.on_transcription_message

    def filtered(message):
        sender = message.get("participantId", "")
        is_final = message.get("rawResponse", {}).get("is_final", False)
        text = message.get("text", "")

        if is_final:
            logger.debug(f"[{name}] transcript: sender={sender}, self={self_id}, text={text[:60]}")

        # Drop self-echo.
        if self_id and sender == self_id[0]:
            return

        # Optional name gate.
        if name_filter and not _name_in_text(name, text):
            if is_final:
                logger.debug(f"[{name}] dropping (not addressed): {text[:60]}")
            return

        orig(message)

    transport._client.on_transcription_message = filtered
    return self_id


def _noop_start_transcription(transport: DailyTransport):
    """Prevent the second agent from calling start_transcription.

    Only one participant may start it; the second attempt raises
    UserMustBeAdmin.  The second agent still receives transcription
    events once the first agent starts it.
    """
    async def _noop(*a, **kw):
        return None
    transport._client.start_transcription = _noop


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_agent(
    *,
    room_url: str,
    token: str,
    name: str,
    system_prompt: str,
    voice_id: str,
    goes_first: bool = False,
    name_filter: bool = False,
    known_agents: set[str] | None = None,
    max_turns: int = 20,
):
    """Join *room_url* as *name* and run an LLM voice pipeline."""

    # -- Transport (audio-only, auto-subscribe) --
    transport = DailyTransport(
        room_url, token, name,
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            transcription_enabled=True,
        ),
    )
    if not goes_first:
        _noop_start_transcription(transport)
    _patch_app_message_filter(transport)

    # -- Services --
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
    tts = ElevenLabsTTSService(api_key=os.getenv("ELEVENLABS_API_KEY"), voice_id=voice_id)

    # -- Context & aggregators --
    msgs = [{"role": "system", "content": system_prompt}]
    ctx = LLMContext(msgs)
    user_agg, asst_agg = LLMContextAggregatorPair(
        ctx,
        user_params=LLMUserAggregatorParams(
            # 2 s silence threshold — prevents false triggers in bot-to-bot rooms
            # where TTS pauses ~1 s between sentences.
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=2.0)),
        ),
    )

    # -- Processors --
    trimmer = ContextTrimmer(msgs, max_turns=10)
    limiter = ConversationLimiter(max_turns=max_turns)

    # -- Pipeline --
    pipeline = Pipeline([
        transport.input(),
        user_agg,
        llm,
        LLMTextRelay(),
        tts,
        transport.output(),
        asst_agg,
        trimmer,
        limiter,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=False,
        ),
        observers=[MetricsLogObserver()],
        rtvi_observer_params=RTVIObserverParams(
            bot_llm_enabled=False,
            bot_tts_enabled=False,
            bot_output_enabled=False,
            bot_speaking_enabled=False,
            user_llm_enabled=False,
            user_speaking_enabled=False,
            user_transcription_enabled=False,
            metrics_enabled=False,
        ),
    )
    limiter.set_task(task)

    # -- end_conversation tool (goes_first agent only) --
    if goes_first:
        end_fn = FunctionSchema(
            name="end_conversation",
            description=(
                "Call this when the conversation has reached a natural "
                "conclusion — both parties have wrapped up, agreed on "
                "next steps, or said goodbye. Do NOT call this while "
                "the conversation is still ongoing."
            ),
            properties={},
            required=[],
        )
        ctx.set_tools(ToolsSchema(standard_tools=[end_fn]))

        async def _handle_end_conversation(params):
            logger.info(f"[{name}] end_conversation called — shutting down in 5s")
            await params.result_callback({"status": "ending"})
            await asyncio.sleep(5.0)
            await task.cancel()

        llm.register_function("end_conversation", _handle_end_conversation)

    # -- Transcription filter --
    self_id = _patch_transcription_filter(transport, name, name_filter)

    # -- Event handlers --
    started = False
    agents = known_agents if known_agents else {name}

    def _is_other_agent(pname: str) -> bool:
        return pname in agents and pname != name

    async def _subscribe(pid: str):
        try:
            await transport.capture_participant_transcription(pid)
            logger.info(f"[{name}] subscribed to transcription: {pid}")
        except Exception as e:
            logger.warning(f"[{name}] capture_transcription error: {e}")

    def _resolve_id(t, data=None) -> str | None:
        if data:
            try:
                return data["participants"]["local"]["id"]
            except (KeyError, TypeError):
                pass
        pid = t.participant_id
        return pid if pid else None

    @transport.event_handler("on_joined")
    async def on_joined(t, data):
        nonlocal started
        my_id = _resolve_id(t, data)
        if my_id:
            self_id.clear()
            self_id.append(my_id)
            await _subscribe(my_id)
        logger.info(f"[{name}] joined (self_id={my_id or '(pending)'})")

        for pid, info in t.participants().items():
            if pid in (my_id, "local"):
                continue
            pname = info.get("info", {}).get("userName", pid)
            if not _is_other_agent(pname):
                continue
            logger.info(f"[{name}] found agent: {pname}")
            await _subscribe(pid)
            if goes_first and not started:
                started = True
                _kick(msgs, task, name)

    @transport.event_handler("on_first_participant_joined")
    async def on_first(t, p):
        nonlocal started
        my_id = _resolve_id(t)
        if my_id and not self_id:
            self_id.append(my_id)

        pname = p.get("info", {}).get("userName", "?")
        if not _is_other_agent(pname):
            return
        logger.info(f"[{name}] first_participant_joined: {pname}")
        await _subscribe(p["id"])
        if goes_first and not started:
            started = True
            _kick(msgs, task, name)

    @transport.event_handler("on_participant_joined")
    async def on_join(t, p):
        nonlocal started
        pname = p.get("info", {}).get("userName", "?")
        if not _is_other_agent(pname):
            return
        logger.info(f"[{name}] participant_joined: {pname}")
        await _subscribe(p["id"])
        if goes_first and not started:
            started = True
            _kick(msgs, task, name)

    @transport.event_handler("on_participant_left")
    async def on_leave(t, p, reason):
        pname = p.get("info", {}).get("userName", "?")
        logger.info(f"[{name}] participant left: {pname} (reason={reason})")
        if _is_other_agent(pname):
            logger.info(f"[{name}] other agent left — shutting down")
            await task.cancel()

    # -- Run --
    runner = PipelineRunner()
    await runner.run(task)


def _kick(msgs: list, task: PipelineTask, name: str):
    """Inject a 'go' system message and trigger the first LLM run."""
    logger.info(f"[{name}] kicking off conversation")
    msgs.append({
        "role": "system",
        "content": "The other person is on the line. Introduce yourself and begin.",
    })
    asyncio.ensure_future(task.queue_frames([LLMRunFrame()]))
