"""Pipeline component factories.

The only module that names a vendor. agent.py asks for "an STT", not "OpenAI's
STT", so swapping a provider (Task 1.2) is a one-line env change and touches
nothing else. Model names and quality/latency settings are pinned decisions --
see tuning.py for each one's rationale.

Plugins are imported at module level on purpose: LiveKit registers each plugin
as a side effect of import, and registration must happen on the main thread.
Importing lazily inside the factories crashes, because the entrypoint runs in a
job worker thread.
"""

from livekit.agents import llm as llm_core
from livekit.agents import tts as tts_core
from livekit.plugins import elevenlabs, google, openai

import config
import persona
import tuning


def build_stt():
    # Single implementation today; this factory is the seam where a second
    # STT vendor would slot in.
    return openai.STT(
        model=tuning.STT_MODEL,
        language=config.STT_LANGUAGE,
        use_realtime=tuning.STT_STREAMING,
        prompt=persona.STT_HINTS,
    )


def _openai_llm():
    return openai.LLM(
        model=tuning.LLM_MODEL_OPENAI,
        temperature=tuning.LLM_TEMPERATURE,
        max_completion_tokens=tuning.LLM_MAX_TOKENS,
    )


def _google_llm():
    return google.LLM(
        model=tuning.LLM_MODEL_GOOGLE,
        api_key=config.GOOGLE_API_KEY,
        temperature=tuning.LLM_TEMPERATURE,
        max_output_tokens=tuning.LLM_MAX_TOKENS,
        thinking_config={"thinking_level": tuning.GEMINI_THINKING_LEVEL},
    )


def build_llm():
    """Primary LLM per config, the other vendor as automatic fallback.

    llm.FallbackAdapter fails over mid-session when the primary errors or
    times out, so one provider outage degrades latency, not the call.
    """
    if config.LLM_PROVIDER == "google":
        chain = [_google_llm(), _openai_llm()]
    elif config.LLM_PROVIDER == "openai":
        chain = [_openai_llm()] + ([_google_llm()] if config.GOOGLE_API_KEY else [])
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")

    # attempt_timeout must be >=10s: Gemini rejects request deadlines under
    # 10s outright (400), which would silently fail every primary call over
    # to OpenAI. Non-timeout errors still fail over immediately.
    return chain[0] if len(chain) == 1 else llm_core.FallbackAdapter(chain, attempt_timeout=10.0)


def build_tts():
    """Primary TTS per config, the other vendor as automatic fallback.

    Failover changes the audible voice mid-call -- accepted: a different
    voice beats a dead one.
    """
    el = (
        elevenlabs.TTS(
            model=tuning.ELEVENLABS_TTS_MODEL,
            voice_id=tuning.ELEVENLABS_TTS_VOICE,
            api_key=config.ELEVENLABS_API_KEY,
            streaming_latency=tuning.ELEVENLABS_STREAMING_LATENCY,
            auto_mode=tuning.ELEVENLABS_AUTO_MODE,
        )
        if config.ELEVENLABS_API_KEY
        else None
    )
    oa = openai.TTS(model=tuning.OPENAI_TTS_MODEL, voice=tuning.OPENAI_TTS_VOICE)

    if config.TTS_PROVIDER == "elevenlabs":
        chain = [el, oa] if el else [oa]
    elif config.TTS_PROVIDER == "openai":
        chain = [oa] + ([el] if el else [])
    else:
        raise ValueError(f"Unknown TTS_PROVIDER: {config.TTS_PROVIDER}")

    return chain[0] if len(chain) == 1 else tts_core.FallbackAdapter(chain)
