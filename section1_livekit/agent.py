"""Entrypoint: assemble the pipeline and run it.

Composition only -- persona in persona.py, tools in tools.py, vendors in
providers.py, settled numbers in tuning.py.

    python agent.py console    # talk to it locally via mic + speakers
    python agent.py dev        # connect to a LiveKit room
"""

import logging
import warnings
from pathlib import Path

# We knowingly use the deprecated turn_detector plugin: it runs locally,
# while its replacement routes through LiveKit's cloud gateway and needs a
# LiveKit account.
warnings.filterwarnings(
    "ignore", message=".*livekit.plugins.turn_detector.*", category=DeprecationWarning
)

from livekit.agents import AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import config
import logs
import providers
import tuning
import webview
from persona import SupportAgent, get_greeting_instructions

# Everything also lands in session.log, overwritten per run (truncated in
# __main__). Handlers append because console mode runs two processes.
LOG_FILE = Path(__file__).parent / "session.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="a")],
)
logger = logging.getLogger("bolt-bites")


async def entrypoint(ctx: JobContext) -> None:
    # Stop here on a misconfigured .env, before any plugin is built.
    config.validate()

    llm_model = (
        tuning.LLM_MODEL_GOOGLE if config.LLM_PROVIDER == "google" else tuning.LLM_MODEL_OPENAI
    )
    logger.info(
        "pipeline: stt=openai llm=%s(%s) tts=%s",
        config.LLM_PROVIDER,
        llm_model,
        config.TTS_PROVIDER,
    )

    session = AgentSession(
        stt=providers.build_stt(),
        llm=providers.build_llm(),
        tts=providers.build_tts(),
        # Local VAD drives both turn-taking and barge-in.
        vad=silero.VAD.load(
            min_silence_duration=tuning.VAD_MIN_SILENCE,
            activation_threshold=tuning.VAD_ACTIVATION,
        ),
        turn_handling={
            # Semantic end-of-turn detection (local ONNX) reads the transcript
            # to judge whether the caller is done -- lets the endpointing
            # delay stay short without barging in mid-sentence.
            "turn_detection": MultilingualModel(),
            "endpointing": {"min_delay": tuning.ENDPOINTING_MIN_DELAY},
            # Start LLM+TTS on the interim transcript while the caller is
            # still finishing; discarded if the final transcript differs.
            "preemptive_generation": {"enabled": True},
        },
    )
    # Narrates the turn story (HEARD / TURN DISCARDED / BARGE-IN / CUT OFF)
    # into session.log so every run explains its own behavior, and mirrors
    # the same events to a live page at http://localhost:8899.
    logs.attach(session)
    await webview.start(session)

    await session.start(room=ctx.room, agent=SupportAgent())
    # The spoken-text tee needs the transcript sink, which exists only now.
    webview.attach_transcript(session)
    await session.generate_reply(instructions=get_greeting_instructions())


if __name__ == "__main__":
    # Fresh log per run; only the parent process reaches __main__.
    LOG_FILE.write_text("")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
