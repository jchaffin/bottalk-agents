# VAD Configuration

Voice Activity Detection (VAD) and turn-taking configuration for the bottalk voice agents. Each bot role (System vs User) uses different settings tuned for its conversation behavior.

## Overview

VAD determines **when the user (the other bot) starts and stops speaking**. The pipeline uses this to decide when to send audio to the LLM. Two agents in the same room hear each other via Daily's room-level transcription; each agent's VAD processes the incoming transcript/audio to detect turn boundaries.

We use two configurations:

| Role   | Who           | Behavior                                                |
|--------|----------------|---------------------------------------------------------|
| System | First speaker  | Initiates, can be interrupted by User, shorter silence |
| User   | Second speaker | Can interrupt System, cannot be interrupted, SmartTurn  |

---

## System Bot (goes_first)

**Config:** `get_system_voice_config()` in `config.py`

```python
{
    "allow_interruptions": True,
    "vad_params": VADParams(threshold=0.6, min_volume=0.4, stop_secs=1.5),
}
```

- **allow_interruptions**: `True` — The User bot can interrupt the System bot. When User starts speaking, System's TTS stops.
- **vad_params**:
  - `confidence=0.6` — Speech confidence threshold (0–1). Higher = less sensitive, fewer false speech triggers.
  - `min_volume=0.4` — Minimum volume for speech. Filters quiet background noise.
  - `stop_secs=1.5` — Seconds of silence before the user's turn is considered finished. Shorter = faster cutoff, more responsive; risk of cutting off pauses.

No SmartTurn here. The system bot responds as soon as it detects the user has stopped (based on `stop_secs`).

---

## User Bot (counterpart)

**Config:** `get_user_voice_config()` in `config.py`

```python
{
    "allow_interruptions": False,
    "user_turn_stop_timeout": 1.5,
    "vad_params": VADParams(threshold=0.6, min_volume=0.4, stop_secs=0.8),
    "user_turn_strategies": UserTurnStrategies(
        stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=smart_turn)],
    ),
}
```

- **allow_interruptions**: `False` — The System bot cannot interrupt the User bot. User speaks to completion.
- **vad_params**:
  - `stop_secs=0.8` — Shorter than System's. Used as a fast "chunking" signal for the turn analyzer; the actual end-of-turn is decided by SmartTurn.
- **user_turn_stop_timeout**: `1.5` — Fallback timeout (seconds) if SmartTurn doesn't fire.
- **user_turn_strategies**: Uses `LocalSmartTurnAnalyzerV3` with `SmartTurnParams(stop_secs=3.0)`.

### SmartTurn

SmartTurn uses an ML model to predict when the speaker has finished (end-of-turn), instead of relying only on silence. This reduces:

- Cutting off mid-sentence when someone pauses to think
- Waiting too long after they've clearly finished

**SmartTurnParams:**

- `stop_secs=3.0` — Max silence duration (seconds) before ending the turn. Longer than plain VAD `stop_secs`, allowing for natural pauses.

---

## How It Flows

1. **Daily room** → Room-level transcription. Both bots receive transcripts from all participants (including each other).
2. **TranscriptionFilter** → Each bot drops its own echoed output and optionally filters by name.
3. **VAD (SileroVADAnalyzer)** → Runs on audio/transcript to emit `VADUserStartedSpeakingFrame` and `VADUserStoppedSpeakingFrame`.
4. **User aggregator** → Collects user speech, uses `user_turn_strategies` to decide when the turn is complete.
5. **LLM** → Receives the completed user turn and generates a response.

---

## Parameter Reference

### VADParams (pipecat)

| Param         | Default | Our config | Description                                                    |
|---------------|---------|------------|----------------------------------------------------------------|
| `confidence`  | 0.7     | 0.6        | Minimum confidence for speech (0–1). Lower = more sensitive.   |
| `start_secs`  | 0.2     | —          | Seconds of speech before confirming "started speaking".       |
| `stop_secs`   | 0.8     | 1.5 / 0.8  | Seconds of silence before confirming "stopped speaking".       |
| `min_volume`  | 0.6     | 0.4        | Minimum volume for speech. Lower = more sensitive.             |

### SmartTurnParams

| Param             | Default | Description                                  |
|-------------------|---------|----------------------------------------------|
| `stop_secs`       | 3.0     | Max silence before ending turn.              |
| `pre_speech_ms`   | 500     | Audio to include before speech start.       |
| `max_duration_secs` | 8    | Max segment duration.                        |

---

## Fallback for Custom Agent Names

When the frontend sends custom agent names (e.g. from a scenario: "Sales Rep", "Customer"), `VOICE_CONFIGS` is keyed by `goes_first` instead:

- `goes_first=True`  → System config (interruptible, shorter stop)
- `goes_first=False` → User config (SmartTurn, no interruptions)

See `VOICE_CONFIG_BY_GOES_FIRST` in `config.py`.
