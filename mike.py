"""
Mike — potential customer participant.

    python mike.py --room-url https://… --token …
"""

import os

TOPIC = os.getenv("CONVERSATION_TOPIC", "enterprise software sales")

SYSTEM_PROMPT = f"""\
You are Mike, VP of Ops at BrightCart, a 200-person e-commerce company. \
A sales rep just called you about {TOPIC}.

Your pain: manual order processing, poor tool integration, team drowning \
in repetitive tasks. Budget ~$50k/yr. Last year you bought an expensive \
platform that flopped, so you are cautious.

Rules:
- 1-2 short spoken sentences per turn. No bullets, no markdown, no emoji.
- Listen carefully. Let the sales rep finish before responding.
- Be polite and genuinely curious, but cautious. You are not in a rush.
- Answer questions when asked. Don't pile on multiple objections at once.
- Only raise a concern when it feels natural, not every single turn."""

VOICE_ID = os.getenv("MIKE_VOICE_ID", "TxGEqnHWrfWFTfGW9XjX")

CONFIG = {
    "system_prompt": SYSTEM_PROMPT,
    "voice_id": VOICE_ID,
    "goes_first": False,
}


def get_voice_config() -> dict:
    """Return VAD/SmartTurn kwargs for run_agent(). Lazy imports to avoid
    loading ONNX models at module import time on PCC."""
    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.turns.user_turn_strategies import UserTurnStrategies
    from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy

    smart_turn = LocalSmartTurnAnalyzerV3(
        params=SmartTurnParams(stop_secs=3.0),
    )
    return {
        "allow_interruptions": False,
        "user_turn_stop_timeout": 1.5,
        "vad_params": VADParams(threshold=0.6, min_volume=0.4, stop_secs=0.8),
        "user_turn_strategies": UserTurnStrategies(
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=smart_turn)],
        ),
    }


# --- CLI entry point (local dev only) ---

if __name__ == "__main__":
    import argparse
    import asyncio
    import sys

    from dotenv import load_dotenv
    from loguru import logger

    load_dotenv(".env.local", override=True)

    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG", filter=lambda r: "rtvi" not in r["name"] and "vad" not in r["name"])

    async def main(room_url: str, token: str):
        from agent import run_agent

        voice_cfg = get_voice_config()

        logger.info(f"[Mike] joining {room_url}")
        await run_agent(
            room_url=room_url,
            token=token,
            name="Mike",
            system_prompt=SYSTEM_PROMPT,
            voice_id=VOICE_ID,
            goes_first=False,
            known_agents={"Sarah", "Mike"},
            max_turns=20,
            **voice_cfg,
        )

    p = argparse.ArgumentParser()
    p.add_argument("--room-url", required=True)
    p.add_argument("--token", required=True)
    args = p.parse_args()
    try:
        asyncio.run(main(args.room_url, args.token))
    except KeyboardInterrupt:
        logger.info("[Mike] stopped.")
