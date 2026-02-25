# bottalk-agents

Pipecat voice agents for [bottalk](https://github.com/jchaffin/bottalk): two AI agents join a shared Daily WebRTC room and role-play scenarios (sales calls, support, discovery) in real time using GPT-4o and ElevenLabs TTS.

---

## Overview

### How bottalk works

1. **Frontend** (Next.js) creates a Daily room, resolves agent prompts from a scenario or custom topic, and calls the Pipecat Cloud start API twice — once per agent.
2. **PCC** (or local dev server) spawns two agent processes, each joining the same room with its own `room_url` and `token`.
3. **Each agent** runs a Pipecat pipeline: audio in (via Daily) → VAD → transcription filter → LLM → TTS → audio out.
4. **Daily** provides room-level Deepgram transcription, so both agents "hear" each other through shared transcript events. The browser joins as a listener and displays a live transcript.

### Why two separate agents?

bottalk simulates **bot-to-bot** conversations. Each agent is an independent Pipecat pipeline with its own prompt, voice, and turn-taking config. The frontend orchestrates by starting both sessions with the same `room_url` and passing the other agent's name in `known_agents`.

### Key constraint: self-echo

Daily broadcasts every participant's speech to all participants. Without filtering, each bot would hear its own TTS echoed back and loop infinitely. A `TranscriptionFilter` in each pipeline drops transcription frames whose `user_id` matches this bot's participant ID, preventing self-echo.

---

## Architecture

### Pipeline flow (per agent)

```
Daily transport (audio in)
    → VAD (Silero) — detect when the other bot starts/stops speaking
    → TranscriptionFilter — drop self-echo, optionally filter by name
    → LLMContext aggregator — buffer user turns
    → OpenAI LLM (GPT-4o)
    → ElevenLabs TTS
    → Daily transport (audio out)
```

### Session arguments (`args.body`)

When PCC or the local dev server starts an agent, it receives:

| Field | Description |
|-------|-------------|
| `room_url` | Daily room URL (both agents share the same room) |
| `token` | Daily meeting token for this participant |
| `name` | Display name (e.g. "Sarah", "Mike") — used for VAD config and transcript display |
| `system_prompt` | Full system prompt (scenario + rules) |
| `voice_id` | ElevenLabs voice ID |
| `goes_first` | `true` = this agent speaks first; `false` = waits for the other |
| `known_agents` | List of all agent names in the room (for `end_conversation` tool) |
| `max_turns` | Hard limit on LLM response turns (default 20) |

The frontend creates the room itself, so PCC does **not** use `createDailyRoom`. All config comes from `body`.

### goes_first and VAD

- **goes_first = true** (System): Starts the conversation, can be interrupted, simpler VAD (silence-based turn end).
- **goes_first = false** (User): Waits for System to speak first, cannot be interrupted, uses **SmartTurn** (LLM-based turn detection) for more accurate end-of-turn.

`config.py` defines `VOICE_CONFIGS` and `VOICE_CONFIG_BY_GOES_FIRST` — VAD params, interruption behavior, and SmartTurn. See `VAD_CONFIG.md` for tuning details.

---

## Files

| File | Purpose |
|------|---------|
| **bot.py** | PCC entry point. Exposes `async def bot(args)`; reads `args.body`, builds pipeline kwargs, calls `run_agent`. Registers `/metrics`, `/metrics/summary`, `/kpis` on PCC Session API. |
| **agent.py** | Core pipeline: `run_agent()` builds Pipeline with Daily transport, Silero VAD, TranscriptionFilter, LLM, TTS, LatencyMetricsObserver. Handles `end_conversation` tool, ConversationLimiter, kickoff. |
| **config.py** | Default prompts (System/User Quick Start), voice IDs, VAD/SmartTurn configs. `AGENT_CONFIGS`, `VOICE_CONFIGS`, `VOICE_CONFIG_BY_GOES_FIRST`. |
| **latency.py** | `LatencyMetricsObserver` — collects per-turn TTFB, LLM, TTS, e2e latency; broadcasts via Daily app-messages; appends to in-memory store for `/metrics`. Optional: POST to `METRICS_INGEST_URL`. |
| **dev.py** | Local dev server (FastAPI). Proxies `/api/start` to create a Daily room and spawn two subprocess agents via `bot.py`. No PCC required. |
| **deploy.py** | Deploy/update agent definition on PCC via REST API. Reads `pcc-deploy.toml`, supports `--image` override for CI. |
| **pcc-deploy.toml** | PCC config: image, secret set, scaling, region, agent profile. |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment

Create `.env.local`:

```env
OPENAI_API_KEY=
ELEVENLABS_API_KEY=
DAILY_API_KEY=           # For app-messages, local dev room creation
PCC_PRIVATE_KEY=         # Deploy only; Pipecat Cloud Dashboard > API Keys > Private
METRICS_INGEST_URL=      # Optional: POST per-turn metrics to your API
```

---

## Local development

```bash
python dev.py
```

Starts FastAPI on port 8000. The bottalk frontend uses `NEXT_PUBLIC_API_URL=http://localhost:8000` to call `/api/start` instead of PCC. `dev.py` creates the Daily room, spawns two `bot.py` subprocesses, and returns room URL/token to the frontend.

---

## Deploy to Pipecat Cloud

### 1. Secrets

Create the `bottalk-secrets` set in PCC (Dashboard or CLI):

```bash
pipecat cloud secrets set bottalk-secrets \
  OPENAI_API_KEY="sk-..." \
  ELEVENLABS_API_KEY="..." \
  DAILY_API_KEY="..." \
  --region us-west
```

### 2. Image pull (private GHCR)

```bash
pipecat cloud secrets image-pull-secret ghcr-credentials https://ghcr.io
# Enter GitHub username and PAT (read:packages) when prompted
```

### 3. Deploy

```bash
python deploy.py
python deploy.py --image ghcr.io/jchaffin/bottalk-agents:abc123  # Override for CI
```

Config: `pcc-deploy.toml` (agent name, image, secret set, scaling, region).

---

## CI

Pushes to `master` trigger a GitHub Action that:

1. Builds the image for `linux/arm64` (with layer cache)
2. Pushes to `ghcr.io/jchaffin/bottalk-agents`
3. Runs `deploy.py --image $IMAGE`

---

## Metrics & observability

- **In-session**: `LatencyMetricsObserver` emits per-turn metrics via Daily app-messages. The frontend subscribes and displays live latency in the transcript.
- **PCC Session API**: `GET /v1/public/{agent}/sessions/{sessionId}/metrics` returns aggregated TTFB, LLM, TTS, e2e, user_bot_latency.
- **Optional**: Set `METRICS_INGEST_URL` to POST per-turn records to your backend for `SessionMetric` storage.

---

## Further reading

- [VAD_CONFIG.md](./VAD_CONFIG.md) — VAD and SmartTurn tuning, System vs User config
- [Pipecat Cloud docs](https://docs.pipecat.ai/deployment/pipecat-cloud/)
- [Daily REST API](https://docs.daily.co/reference/rest-api)
