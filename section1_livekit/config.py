"""Environment-backed settings: secrets and per-run switches only.

Settled numeric/model decisions live in tuning.py with their rationale.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Vendor switches -- each has >=2 working implementations in providers.py, so
# swapping is a config change, never a code change (Task 1.2).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs")

# Keys. OPENAI_API_KEY is read from the environment by the plugins directly;
# these two are passed explicitly (the ElevenLabs plugin looks for a
# differently-named var, and Google would fall back to gcloud auth).
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# Language mode. Set both to "ar" for the Egyptian Arabic persona.
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")
PERSONA_LANG = os.getenv("PERSONA_LANG", "en")

# Demo dials: fake backend latency (ms) inside the tools, so the filler race
# is observable against the instant mock store. 0 = off (production: the real
# backend supplies the delay). Cancels get their own knob -- writes are
# slower than reads in real backends.
TOOL_LATENCY_MS = int(os.getenv("TOOL_LATENCY_MS", "0"))
CANCEL_LATENCY_MS = int(os.getenv("CANCEL_LATENCY_MS", str(TOOL_LATENCY_MS)))


def validate() -> None:
    """Fail fast with a clear one-liner if a required key is missing.

    Called before providers are built, so a misconfigured .env stops here
    instead of deep inside a plugin with a cryptic auth error.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing -- set it in .env")
    if TTS_PROVIDER == "elevenlabs" and not ELEVENLABS_API_KEY:
        raise SystemExit("TTS_PROVIDER=elevenlabs but ELEVENLABS_API_KEY is missing")
    if LLM_PROVIDER == "google" and not GOOGLE_API_KEY:
        raise SystemExit("LLM_PROVIDER=google but GOOGLE_API_KEY is missing")
