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
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.daily.transport import DailyParams, DailyTransport


class ContextTrimmer(FrameProcessor):
    """Keep the LLM context bounded to a sliding window of recent turns.

    Without trimming, ``msgs`` grows every turn and GPT-4o's
    time-to-first-token climbs steadily — after enough turns the gap
    between audio playback and new transcript text becomes very
    noticeable.

    Placed after ``asst_agg`` in the pipeline so it sees
    ``LLMFullResponseEndFrame`` once the assistant message has been
    appended to ``msgs``.
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
        system = [m for m in self._messages if m.get("role") == "system"]
        convo = [m for m in self._messages if m.get("role") != "system"]

        limit = self._max_turns * 2  # user + assistant per turn
        if len(convo) > limit:
            self._messages[:] = system + convo[-limit:]
            logger.debug(
                f"Trimmed context: kept {len(system)} system + {limit} convo "
                f"msgs (dropped {len(convo) - limit})"
            )


class LLMTextRelay(FrameProcessor):
    """Send batched ``bot-llm-text`` app-messages as LLM tokens stream.

    Each Daily ``send_app_message`` awaits an acknowledgement future.
    Sending one message per LLM token (~50/sec) means the transport's
    input-task spends most of its time blocked on ack futures, creating
    an ever-growing backlog that makes the transcript fall exponentially
    behind the audio.

    Fix: buffer tokens and flush every ``_FLUSH_CHARS`` characters
    (~100 chars ≈ 2-3 msgs/sec instead of 50).  ``LLMFullResponseEndFrame``
    always triggers a final flush so no text is lost.
    """

    _FLUSH_CHARS = 100

    def __init__(self):
        super().__init__()
        self._buffer = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = ""
            await self.push_frame(
                OutputTransportMessageUrgentFrame(
                    message={"label": "outrival", "type": "bot-llm-started"}
                )
            )
        elif isinstance(frame, LLMTextFrame):
            self._buffer += frame.text
            if len(self._buffer) >= self._FLUSH_CHARS:
                await self._flush_buffer()
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._flush_buffer()
            await self.push_frame(
                OutputTransportMessageUrgentFrame(
                    message={"label": "outrival", "type": "bot-llm-stopped"}
                )
            )

        await self.push_frame(frame, direction)

    async def _flush_buffer(self):
        if self._buffer:
            text = self._buffer
            self._buffer = ""
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


def _patch_app_message_filter(transport: DailyTransport):
    """Drop incoming ``outrival``-labelled app-messages before RTVI sees them.

    When two agents share a room, each agent's ``LLMTextRelay`` sends
    custom app-messages (``bot-llm-text``, etc.) via Daily's broadcast
    ``send_app_message``.  The PCC base image wraps every agent with an
    RTVI processor that validates ALL incoming transport messages.  Our
    custom messages fail RTVI validation, flooding the event loop with
    errors and choking the pipeline so the agent never responds.

    Fix: intercept ``on_app_message`` on the low-level Daily client and
    silently drop messages with ``label == "outrival"`` before they enter
    the pipeline.  The browser client still receives them (it listens
    via its own Daily call object, not through pipecat).
    """
    _orig = transport._client.on_app_message

    def _filtered(message, sender):
        if isinstance(message, dict) and message.get("label") == "outrival":
            return
        _orig(message, sender)

    transport._client.on_app_message = _filtered


def _patch_transcription_filter(
    transport: DailyTransport,
    name: str,
    name_filter: bool,
) -> list[str]:
    """Patch the transport to filter out self-transcription and optionally gate by name.

    ``DailyTransport.__init__`` captures bound-method references in a
    ``DailyCallbacks`` object, so patching ``transport._on_transcription_message``
    after init has no effect.  The Daily SDK calls the sync method
    ``DailyTransportClient.on_transcription_message`` directly, so we patch
    that on the internal ``_client`` instance instead.

    Returns a mutable list used to store the bot's own participant ID.
    Event handlers should populate it (via ``.append``) once the ID is known.
    """
    # Mutable container so event handlers can later store the bot's participant
    # ID via self_id.append(my_id).  Using a list (rather than a plain str)
    # lets the inner closure see updates without requiring ``nonlocal``.
    self_id: list[str] = []

    # Save a reference to the original (unpatched) handler so we can
    # delegate to it for messages that pass our filters.
    _orig_handler = transport._client.on_transcription_message

    def _filtered_handler(message):
        """Drop transcriptions from the bot itself (and optionally from
        anyone not addressing the bot by name), then forward the rest
        to the original Daily handler."""

        sender = message.get("participantId", "")
        is_final = message.get("rawResponse", {}).get("is_final", False)
        text = message.get("text", "")

        # Log final transcriptions for debugging (partials are too noisy).
        if is_final:
            logger.debug(
                f"[{name}] transcription filter: sender={sender}, self={self_id}, text={text[:60]}"
            )

        # --- Self-transcription gate ---
        # If we already know our own ID and this message came from us,
        # drop it to prevent the feedback loop.
        if self_id and sender == self_id[0]:
            return

        # --- Name-based gating (optional) ---
        # When enabled, only forward messages that mention the bot's name
        # as a whole word.  We check *all* messages (partial + final) so
        # that unaddressed speech never reaches the aggregator / VAD pipeline.
        if name_filter:
            if not _name_in_text(name, text):
                if is_final:
                    logger.debug(f"[{name}] dropping (not addressed): {text[:60]}")
                return

        # Message passed all filters — hand it off to the real handler.
        _orig_handler(message)

    # Swap in our filtering wrapper on the low-level client.
    transport._client.on_transcription_message = _filtered_handler
    return self_id


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
):
    """Join *room_url* as *name* and run an LLM voice pipeline.

    When *name_filter* is True, only final transcription messages that
    contain the bot's *name* (whole-word, case-insensitive) are forwarded
    to the pipeline.  This lets a human user address bots individually.
    """

    # -- Transport --
    # Both agents need transcription_enabled=True so that
    # capture_participant_transcription() actually works (the pipecat
    # implementation early-returns when it's False).
    #
    # However, only ONE participant may call Daily's start_transcription
    # — a second attempt fails with UserMustBeAdmin.  So for the agent
    # that does NOT go first we monkey-patch start_transcription to a
    # no-op; it will still receive transcription events once the first
    # agent starts it (Daily broadcasts transcription-started to all).
    transport = DailyTransport(
        room_url, token, name,
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            transcription_enabled=True,
        ),
    )

    if not goes_first:
        async def _noop_start(*a, **kw):
            return None
        transport._client.start_transcription = _noop_start

    # -- Drop custom app-messages before RTVI validation --
    _patch_app_message_filter(transport)

    # -- Services --
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
    tts = ElevenLabsTTSService(api_key=os.getenv("ELEVENLABS_API_KEY"), voice_id=voice_id)

    # -- LLM context & aggregators --
    # `msgs` is the running conversation history shared with the LLM.
    # The aggregator pair splits incoming frames into user vs. assistant
    # turns and feeds them into `ctx`.  VAD (Voice Activity Detection)
    # is used on the user side to detect when the human stops speaking.
    msgs = [{"role": "system", "content": system_prompt}]
    ctx = LLMContext(msgs)
    # stop_secs: how long to wait after silence before declaring the other
    # speaker has finished.  Default 0.8 s is fine for human → bot, but in
    # a bot-to-bot room the TTS can pause ~1 s between sentences, causing
    # the listener to jump in mid-response.  2 s eliminates false triggers.
    user_agg, asst_agg = LLMContextAggregatorPair(
        ctx,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=2.0)),
        ),
    )

    # -- Context trimmer --
    # After each completed response, drop old turns from the message
    # history so the LLM prompt stays bounded.  Without this, TTFT
    # grows linearly with conversation length and the transcript text
    # visibly lags behind the TTS audio after ~10 turns.
    trimmer = ContextTrimmer(msgs, max_turns=10)

    # -- Pipeline --
    # Frames flow left-to-right:
    #   audio in → user aggregator → LLM → relay → TTS → audio out
    #   → assistant aggregator → context trimmer
    # LLMTextRelay intercepts LLM frames and pushes bot-llm-text messages
    # directly downstream, so they only traverse TTS → transport.output
    # (2 hops) instead of the full RTVI observer path (6+ hops).
    pipeline = Pipeline([
        transport.input(),
        user_agg,
        llm,
        LLMTextRelay(),
        tts,
        transport.output(),
        asst_agg,
        trimmer,
    ])

    task = PipelineTask(pipeline, params=PipelineParams(
        enable_metrics=True,
        enable_usage_metrics=True,
        allow_interruptions=False,
    ),
        # Disable ALL RTVI observer app-messages.  LLMTextRelay already
        # handles bot-llm-text with a shorter pipeline path, and the
        # frontend doesn't use any of the other RTVI message types.
        # Leaving them enabled floods the Daily signaling WebSocket
        # (bot-tts-text alone is dozens of msgs/sec per agent) causing
        # the transcript to fall progressively further behind the audio.
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

    # -- Transcription filter --
    # Intercept raw transcription messages at the transport level to drop
    # self-echoes (and optionally messages not addressed to this bot).
    # Returns a mutable list that event handlers populate with our own
    # participant ID once it becomes available.
    self_id = _patch_transcription_filter(transport, name, name_filter)

    # Tracks whether this bot has already kicked off the conversation
    # (only relevant when ``goes_first=True``).
    started = False

    async def _subscribe(pid: str):
        """Ask Daily to send us transcription events for participant *pid*."""
        try:
            await transport.capture_participant_transcription(pid)
            logger.info(f"[{name}] capture_participant_transcription({pid})")
        except Exception as e:
            logger.warning(f"[{name}] capture_transcription error: {e}")

    def _resolve_id(t, data=None) -> str | None:
        """Best-effort extraction of our own participant ID.

        Tries the ``data`` payload first (available in ``on_joined``),
        then falls back to ``transport.participant_id``.
        """
        if data:
            try:
                return data["participants"]["local"]["id"]
            except (KeyError, TypeError):
                pass
        pid = t.participant_id
        return pid if pid else None

    # -----------------------------------------------------------------
    # Transport event handlers
    #
    # Daily fires these callbacks as participants join / leave the room.
    # Our goals:
    #   1. Record our own participant ID into ``self_id`` so the
    #      transcription filter can drop self-echoes.
    #   2. Subscribe to transcription for every *other* agent in the room.
    #   3. If ``goes_first``, kick off the conversation once another
    #      agent is present.
    # -----------------------------------------------------------------

    # Build the set of agent names we recognise in this room.
    # When not provided, fall back to just our own name (so we still
    # filter self-echoes, but treat every other participant as an agent).
    _agents = known_agents if known_agents else {name}

    def _is_other_agent(pname):
        """True if *pname* is a known agent that isn't us."""
        return pname in _agents and pname != name

    @transport.event_handler("on_joined")
    async def on_joined(t, data):
        """Fires once when *this* bot successfully joins the room."""
        nonlocal started

        # Resolve and store our own participant ID.
        my_id = _resolve_id(t, data)
        if my_id:
            self_id.clear()
            self_id.append(my_id)
            await _subscribe(my_id)
        logger.info(f"[{name}] joined (self_id={my_id or '(pending)'})")

        # Walk through participants already in the room and subscribe to
        # any that are known agents (skipping ourselves).
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
        """Fires when the first *remote* participant joins after us."""
        nonlocal started

        # Fallback: capture our own ID if on_joined didn't provide it.
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
        """Fires for every subsequent remote participant that joins."""
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
        """Log when a participant leaves (no cleanup needed currently)."""
        pname = p.get("info", {}).get("userName", "?")
        logger.info(f"[{name}] participant left: {pname}")

    # -- Run --
    # Block until the pipeline shuts down (room closed, error, etc.).
    runner = PipelineRunner()
    await runner.run(task)


def _kick(msgs, task, name):
    """Append a 'go' system message and queue an ``LLMRunFrame``.

    This nudges the LLM to speak first by injecting a system-level
    instruction into the conversation history and immediately triggering
    a pipeline run.  Called at most once per agent (guarded by
    ``started`` in the event handlers).
    """
    logger.info(f"[{name}] kicking off conversation")
    msgs.append({
        "role": "system",
        "content": "The other person is on the line. Introduce yourself and begin.",
    })
    # Fire-and-forget: schedule the frame push without awaiting it,
    # since this helper is called from a sync context.
    import asyncio
    asyncio.ensure_future(task.queue_frames([LLMRunFrame()]))
