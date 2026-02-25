# Copyright 2026 Jacob Chaffin / bottalk. All rights reserved.

"""
Agent config: default prompts and voice settings for System & User.

Used by bot.py as fallbacks when the API sends empty prompts (Quick Start).
"""

import os
import re

# Matches {{ var: default }} — replaces with default when used as Quick Start fallback
_VAR_DEFAULT_RE = re.compile(r'\{\{\s*\w+\s*:\s*([^}]*)\s*\}\}')

# Default agent display names. Keep in sync with frontend src/lib/config.ts VOICES[0].name, VOICES[1].name.
DEFAULT_AGENT_1_NAME = "Sarah"
DEFAULT_AGENT_2_NAME = "Mike"

SYSTEM_PROMPT = f"""\
You are {{{{ name: {DEFAULT_AGENT_1_NAME} }}}} an {{{{ role: enterprise software sales rep }}}} at {{{{ company: bottalk }}}}. \
You are on a live phone call with a potential customer about {{{{ topic: enterprise software sales }}}}.

Your product — {{{{ company: bottalk }}}} — is an AI workflow automation platform that \
integrates with Salesforce, HubSpot, Slack, Jira, and 50+ other tools. \
Professional tier: $99/user/month. 30-day free trial. Case study: Acme \
Corp cut manual work by 60% in 3 months."""

USER_PROMPT = f"""\
You are {{{{ name: {DEFAULT_AGENT_2_NAME} }}}}, {{{{ role: VP of Ops }}}} at {{{{ company: BrightCart }}}}, a 200-person e-commerce company. \
A sales rep just called you about {{{{ topic: enterprise software sales }}}}.

Your pain: manual order processing, poor tool integration, team drowning \
in repetitive tasks. Budget ~$50k/yr. Last year you bought an expensive \
platform that flopped, so you are cautious.

Rules:
- You are the one being called. Acknowledge the caller and let them proceed (e.g. "Hi, go ahead" or "Sure, what's this about?") — do NOT say "How can I help you?" or offer to help them."""

SYSTEM_VOICE_ID = os.getenv("SYSTEM_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
USER_VOICE_ID = os.getenv("USER_VOICE_ID", "TxGEqnHWrfWFTfGW9XjX")

def _apply_default_vars(text: str) -> str:
    """Replace {{ var: default }} with default when used as Quick Start fallback."""
    return _VAR_DEFAULT_RE.sub(lambda m: m.group(1).strip(), text)


AGENT_CONFIGS = {
    "System": {
        "system_prompt": _apply_default_vars(SYSTEM_PROMPT),
        "voice_id": SYSTEM_VOICE_ID,
        "goes_first": True,
    },
    "User": {
        "system_prompt": _apply_default_vars(USER_PROMPT),
        "voice_id": USER_VOICE_ID,
        "goes_first": False,
    },
}


def get_system_voice_config() -> dict:
    """Return VAD kwargs for System bot."""
    from pipecat.audio.vad.vad_analyzer import VADParams

    return {
        "allow_interruptions": True,
        "vad_params": VADParams(confidence=0.6, min_volume=0.4, stop_secs=1.5),
    }


def get_user_voice_config() -> dict:
    """Return VAD + SmartTurn kwargs for User bot."""
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
    from pipecat.turns.user_turn_strategies import UserTurnStrategies
    from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy

    smart_turn = LocalSmartTurnAnalyzerV3(
        params=SmartTurnParams(stop_secs=1.2),  # 3.0 was too slow; bot↔bot = complete turns
    )

    return {
        "allow_interruptions": False,
        "user_turn_stop_timeout": 1.5,
        "vad_params": VADParams(confidence=0.6, min_volume=0.4, stop_secs=0.8),
        "user_turn_strategies": UserTurnStrategies(
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=smart_turn)],
        ),
    }


VOICE_CONFIGS = {
    "System": get_system_voice_config,
    "User": get_user_voice_config,
}

# Fallback when agent name is custom (e.g. from scenario): use goes_first for VAD config.
VOICE_CONFIG_BY_GOES_FIRST = {
    True: get_system_voice_config,
    False: get_user_voice_config,
}
