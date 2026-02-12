"""
Reusable Pipecat voice agent that joins a Daily room.

Key detail for multi-bot rooms
------------------------------
Daily's room-level Deepgram transcription broadcasts every participant's
speech to ALL participants. Without filtering, each bot "hears" its own
TTS output echoed back as user input → infinite feedback loop.

We fix this by monkey-patching the transport's ``_on_transcription_message``
callback so it silently drops any transcription whose ``participantId``
matches this bot. This is done at the transport level (before frames enter
the pipeline), avoiding FrameProcessor start-up ordering issues.

When ``name_filter=True``, we additionally gate on whether the bot's name
appears as a whole word in the *final* transcription text. This allows a
human user to address individual bots by name in a multi-party room.
"""

import os
import re
from typing import Any, Mapping

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIObserverParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.daily.transport import DailyParams, DailyTransport


class _LLMTextRelay(FrameProcessor):
    """Batched LLM-text relay — sends ``outrival``-labelled app-messages.

    Daily's ``send_app_message`` awaits an ack future for every call.
    Sending per-token (~50/sec) saturates the signalling channel and the
    transcript falls further behind the audio with every turn.

    Batching to ~100 chars drops the rate to ~2-3 msgs/sec — well within
    capacity — while still giving smooth streaming text on screen.
    """

    _FLUSH_CHARS = 100

    def __init__(self):
        super().__init__()
        self._buf = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buf = ""
            await self.push_frame(
                OutputTransportMessageUrgentFrame(
                    message={"label": "outrival", "type": "bot-llm-started"}
                )
            )
        elif isinstance(frame, LLMTextFrame):
            self._buf += frame.text
            if len(self._buf) >= self._FLUSH_CHARS:
                await self._flush()
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._flush()
            await self.push_frame(
                OutputTransportMessageUrgentFrame(
                    message={"label": "outrival", "type": "bot-llm-stopped"}
                )
            )

        await self.push_frame(frame, direction)

    async def _flush(self):
        if self._buf:
            text, self._buf = self._buf, ""
            await self.push_frame(
                OutputTransportMessageUrgentFrame(
                    message={
                        "label": "outrival",
                        "type": "bot-llm-text",
                        "data": {"text": text},
                    }
                )
            )


def _name_in_text(name: str, text: str) -> bool:
    """Return True if *name* appears as a whole word in *text* (case-insensitive)."""
    pattern = r"\b" + re.escape(name) + r"\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


KNOWN_AGENTS = {"Sarah", "Mike"}

async def run_agent(
    *,
    room_url: str,
    token: str,
    name: str,
    system_prompt: str,
    voice_id: str,
    goes_first: bool = False,
    name_filter: bool = False,
):
    """Join *room_url* as *name* and run an LLM voice pipeline.

    When *name_filter* is True, only final transcription messages that
    contain the bot's *name* (whole-word, case-insensitive) are forwarded
    to the pipeline.  This lets a human user address bots individually.
    """

    transport = DailyTransport(
        room_url, token, name,
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            transcription_enabled=True,
        ),
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
    tts = ElevenLabsTTSService(api_key=os.getenv("ELEVENLABS_API_KEY"), voice_id=voice_id)

    msgs = [{"role": "system", "content": system_prompt}]
    ctx = LLMContext(msgs)
    user_agg, asst_agg = LLMContextAggregatorPair(
        ctx, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline([
        transport.input(),
        user_agg,
        llm,
        _LLMTextRelay(),
        tts,
        transport.output(),
        asst_agg,
    ])

    task = PipelineTask(pipeline, params=PipelineParams(
        enable_metrics=True,
        enable_usage_metrics=True,
        allow_interruptions=False,
    ),
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

    # ------------------------------------------------------------------
    # Monkey-patch: filter self-transcription at the transport level
    #
    # DailyTransport.__init__ captures bound-method references in a
    # DailyCallbacks object, so patching transport._on_transcription_message
    # after init has NO effect.  The Daily SDK calls the sync method
    # DailyTransportClient.on_transcription_message directly, so we patch
    # THAT on the internal _client instance.
    # ------------------------------------------------------------------
    self_id: list[str] = []          # mutable container so the closure can update it
    _orig_handler = transport._client.on_transcription_message   # sync method

    def _filtered_handler(message):
        sender = message.get("participantId", "")
        is_final = message.get("rawResponse", {}).get("is_final", False)
        text = message.get("text", "")
        if is_final:
            logger.debug(f"[{name}] transcription filter: sender={sender}, self={self_id}, text={text[:60]}")
        if self_id and sender == self_id[0]:
            return                   # silently drop our own speech

        # Name-based gating: only process transcriptions that mention us.
        # We check *all* messages (partial + final) so that unaddressed
        # speech never reaches the aggregator / VAD pipeline.
        if name_filter:
            if not _name_in_text(name, text):
                if is_final:
                    logger.debug(f"[{name}] dropping (not addressed): {text[:60]}")
                return
        _orig_handler(message)

    transport._client.on_transcription_message = _filtered_handler
    # ------------------------------------------------------------------

    started = False

    async def _subscribe(pid: str):
        try:
            await transport.capture_participant_transcription(pid)
            logger.info(f"[{name}] capture_participant_transcription({pid})")
        except Exception as e:
            logger.warning(f"[{name}] capture_transcription error: {e}")

    def _resolve_id(t, data=None) -> str | None:
        """Best-effort extraction of our own participant ID."""
        if data:
            try:
                return data["participants"]["local"]["id"]
            except (KeyError, TypeError):
                pass
        pid = t.participant_id
        return pid if pid else None

    # --- transport events ---

    def _is_other_agent(pname):
        """True if pname is a known agent that isn't us."""
        return pname in KNOWN_AGENTS and pname != name

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
                logger.debug(f"[{name}] ignoring non-agent: {pname}")
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
            logger.debug(f"[{name}] ignoring non-agent: {pname}")
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
            logger.debug(f"[{name}] ignoring non-agent: {pname}")
            return
        logger.info(f"[{name}] participant_joined: {pname}")
        await _subscribe(p["id"])
        if goes_first and not started:
            started = True
            _kick(msgs, task, name)

    @transport.event_handler("on_participant_left")
    async def on_leave(t, p, reason):
        pname = p.get("info", {}).get("userName", "?")
        logger.info(f"[{name}] participant left: {pname}")

    runner = PipelineRunner()
    await runner.run(task)


def _kick(msgs, task, name):
    """Append the 'go' system message and queue the LLM run frame."""
    logger.info(f"[{name}] kicking off conversation")
    msgs.append({
        "role": "system",
        "content": "The other person is on the line. Introduce yourself and begin.",
    })
    import asyncio
    asyncio.ensure_future(task.queue_frames([LLMRunFrame()]))
