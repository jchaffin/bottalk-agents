"""
Reusable Pipecat voice agent that joins a Daily room.

Multi-bot room handling
-----------------------
Daily's room-level Deepgram transcription broadcasts every participant's
speech to ALL participants.  Without filtering, each bot "hears" its own
TTS output echoed back as user input — infinite feedback loop.

A ``TranscriptionFilter`` FrameProcessor sits in the pipeline between
``transport.input()`` and the user aggregator.  It drops any
``TranscriptionFrame`` / ``InterimTranscriptionFrame`` whose ``user_id``
matches this bot's participant ID.  When ``name_filter=True``, it
additionally requires the bot's name to appear as a whole word in the
transcription text.
"""

import asyncio
import os
import re

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndTaskFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMRunFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver

from latency import LatencyMetricsObserver
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.daily.transport import DailyParams, DailyTransport


KNOWN_AGENTS = {"Sarah", "Mike"}


# ---------------------------------------------------------------------------
# Pipeline processors
# ---------------------------------------------------------------------------

class TranscriptionFilter(FrameProcessor):
    """Drop transcription frames that came from this bot (self-echo) or
    that don't mention this bot's name (when *name_filter* is enabled).

    Works at the pipeline level — no monkey-patching of transport internals.
    """

    def __init__(
        self,
        *,
        name: str,
        self_id: list[str],
        name_filter: bool = False,
    ):
        super().__init__()
        self._name = name
        self._self_id = self_id
        self._name_filter = name_filter

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            sender = frame.user_id
            text = frame.text
            is_final = isinstance(frame, TranscriptionFrame)

            if is_final:
                logger.debug(
                    f"[{self._name}] transcript: sender={sender}, "
                    f"self={self._self_id}, text={text[:60]}"
                )

            # Drop our own speech echoed back via Deepgram.
            if self._self_id and sender == self._self_id[0]:
                return

            # Optionally require the bot's name to appear in the text.
            if self._name_filter and not _name_in_text(self._name, text):
                if is_final:
                    logger.debug(
                        f"[{self._name}] dropping (not addressed): {text[:60]}"
                    )
                return

        await self.push_frame(frame, direction)


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



# ---------------------------------------------------------------------------
# Transport patches
# ---------------------------------------------------------------------------

def _name_in_text(name: str, text: str) -> bool:
    """Check if *name* appears as a whole word (case-insensitive)."""
    return bool(re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE))


def _noop_start_transcription(transport: DailyTransport):
    """Prevent the second agent from calling start_transcription.

    Only one participant may start it; the second attempt raises
    UserMustBeAdmin.  The second agent still receives transcription
    events once the first agent starts it.

    We patch the *DailyTransport* method directly (not the internal
    ``_client``) so the override is guaranteed to be hit by Pipecat's
    join handler.
    """
    async def _noop(*a, **kw):
        return None
    transport.start_transcription = _noop  # type: ignore[assignment]


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
    allow_interruptions: bool = False,
    user_turn_strategies: UserTurnStrategies | None = None,
    user_turn_stop_timeout: float = 5.0,
    vad_params: VADParams | None = None,
    llm_model: str = "gpt-4o-mini",
    session_id: str = "",
    metrics_store: list | None = None,
    kpi_store: dict | None = None,
):
    """Join *room_url* as *name* and run an LLM voice pipeline."""

    if vad_params is None:
        vad_params = VADParams(threshold=0.6, min_volume=0.4, stop_secs=0.2)

    # -- Transport --
    transport = DailyTransport(
        room_url, token, name,
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            transcription_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=vad_params),
        ),
    )
    if not goes_first:
        _noop_start_transcription(transport)

    # -- Services --
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=llm_model,
        params=OpenAILLMService.InputParams(
            max_completion_tokens=200,
            temperature=0.7,
        ),
    )
    tts = ElevenLabsTTSService(api_key=os.getenv("ELEVENLABS_API_KEY"), voice_id=voice_id)

    # -- Context & aggregators --
    msgs = [{"role": "system", "content": system_prompt}]
    ctx = LLMContext(msgs)
    user_params = LLMUserAggregatorParams(
        user_turn_strategies=user_turn_strategies or UserTurnStrategies(),
        user_turn_stop_timeout=user_turn_stop_timeout,
    )
    user_agg, asst_agg = LLMContextAggregatorPair(ctx, user_params=user_params)

    # -- Processors --
    self_id: list[str] = []
    tx_filter = TranscriptionFilter(
        name=name, self_id=self_id, name_filter=name_filter,
    )
    limiter = ConversationLimiter(max_turns=max_turns)

    # -- Pipeline --
    pipeline = Pipeline([
        transport.input(),
        tx_filter,
        user_agg,
        llm,
        tts,
        transport.output(),
        asst_agg,
        limiter,
    ])

    # Extract room name from URL for Daily REST API app-message broadcast
    room_name = room_url.rstrip("/").rsplit("/", 1)[-1] if room_url else ""

    latency_obs = LatencyMetricsObserver(
        agent_name=name,
        session_id=session_id,
        metrics_store=metrics_store,
        room_name=room_name,
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=allow_interruptions,
        ),
        observers=[
            MetricsLogObserver(),
            latency_obs,
        ],
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
            logger.info(f"[{name}] end_conversation called — stopping session")
            await params.result_callback({"status": "ending"})
            await params.llm.push_frame(TTSSpeakFrame("Thank you for the conversation. Goodbye!"))
            try:
                await transport.stop_transcription()
            except Exception:
                pass
            await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

        llm.register_function("end_conversation", _handle_end_conversation)

    # -- Event handlers --
    started = False
    agents = known_agents if known_agents else KNOWN_AGENTS
    subscribed: set[str] = set()

    def _matches_known_agent_name(display_name: str) -> bool:
        """Best-effort match for Daily participant display names.

        Daily's userName can include extra tokens (e.g. "Sarah (bot)"), so we
        match any known agent name as a whole word (case-insensitive).
        """
        if not display_name:
            return False
        for n in agents:
            if n and n != name and _name_in_text(n, display_name):
                return True
        return False

    async def _subscribe(pid: str):
        if not pid or pid in subscribed:
            return
        try:
            await transport.capture_participant_transcription(pid)
            subscribed.add(pid)
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
            await _subscribe(pid)
            if goes_first and not started and _matches_known_agent_name(pname):
                logger.info(f"[{name}] found agent: {pname}")
                started = True
                _kick(msgs, task, name)

    @transport.event_handler("on_first_participant_joined")
    async def on_first(t, p):
        nonlocal started
        my_id = _resolve_id(t)
        if my_id and not self_id:
            self_id.append(my_id)

        pname = p.get("info", {}).get("userName", "?")
        logger.info(f"[{name}] first_participant_joined: {pname}")
        await _subscribe(p["id"])
        if goes_first and not started and _matches_known_agent_name(pname):
            started = True
            _kick(msgs, task, name)

    @transport.event_handler("on_participant_joined")
    async def on_join(t, p):
        nonlocal started
        pname = p.get("info", {}).get("userName", "?")
        logger.info(f"[{name}] participant_joined: {pname}")
        await _subscribe(p["id"])
        if goes_first and not started and _matches_known_agent_name(pname):
            started = True
            _kick(msgs, task, name)
        elif not _matches_known_agent_name(pname):
            # Non-agent participant (browser observer) joined late —
            # re-broadcast full metric history so it can catch up.
            latency_obs.broadcast_history()

    @transport.event_handler("on_participant_left")
    async def on_leave(t, p, reason):
        pname = p.get("info", {}).get("userName", "?")
        logger.info(f"[{name}] participant left: {pname} (reason={reason})")
        if _matches_known_agent_name(pname):
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
