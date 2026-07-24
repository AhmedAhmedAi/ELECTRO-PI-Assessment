"""Settled tuning decisions -- every value here was measured on this pipeline,
not guessed. Format per entry: What it does / Why this value / Other options.

Vendor SWITCHES (which provider) live in .env because swapping vendors with a
config change is the point (Task 1.2). Everything here is a pinned decision:
change it by editing this file, knowingly.
"""

# --- Turn taking (the timers you feel) --------------------------------------

VAD_MIN_SILENCE = 0.35
# What: seconds of continuous mic silence before "the caller stopped talking".
#   This is the floor of the agent's reaction time -- she can never answer
#   faster than this.
# Why 0.35: battle-tested in the author's HOTLY production voice app; the
#   0.55 LiveKit default added 0.2s of dead air per turn with no accuracy gain.
# Options: 0.3 (snappier, starts chopping mid-word pauses) .. 0.55 (default).

VAD_ACTIVATION = 0.5
# What: confidence (0-1) the VAD needs to call sound "voice".
# Why 0.5: the model default; fine in a quiet room.
# Options: 0.6-0.7 in noisy rooms (stops ambient chatter triggering turns,
#   may miss soft speech).

ENDPOINTING_MIN_DELAY = 0.0
# What: extra wait after VAD silence before committing the turn to the LLM.
# Why 0: eager-commit design. turns.py purges speculative lookups if the
#   caller resumes talking, so waiting buys nothing -- the LiveKit default of
#   0.5 assumes no such recovery mechanism exists. The semantic turn detector
#   (see agent.py) remains the brake against mid-sentence commits.
# Options: 0.15-0.3 if dictated numbers still fragment despite the detector.

# --- STT ---------------------------------------------------------------------

STT_MODEL = "gpt-4o-transcribe"
STT_STREAMING = True
# What: stream mic audio over a websocket WHILE the caller speaks, instead of
#   uploading the clip after they stop.
# Why True: measured ~1s saved per turn -- buffered mode starts transcribing
#   from zero after end-of-speech; streaming has the text essentially ready.
#   It also produces interim transcripts, without which preemptive generation
#   (agent.py) has nothing to work on and silently does nothing.
# Options: False only for debugging the buffered path.

# --- LLM ---------------------------------------------------------------------

LLM_MODEL_GOOGLE = "gemini-3.1-flash-lite"
# Why: measured ttft 0.39-0.43s stable from this machine vs OpenAI's
#   0.65-0.87s with jitter spikes to 2.7s. Google serves from nearby edge.
LLM_MODEL_OPENAI = "gpt-4o-mini"
# Why: the fallback when LLM_PROVIDER=openai -- cheapest reliable tool-caller.

GEMINI_THINKING_LEVEL = "minimal"
# What: Gemini 3 is a reasoning model and thinks before answering by default.
# Why minimal: measured 0.69s total vs 1.22s at "low" -- thinking is seconds
#   of dead air on a phone call, and support answers don't need it.
# Options: "low"/"high" if the agent ever needs multi-step reasoning.

LLM_MAX_TOKENS = 120
# What: hard cap on reply length; also caps worst-case speaking time (~9s).
# Why 120: one-to-two spoken sentences. The persona asks for brevity; the cap
#   enforces it when the model rambles anyway.

LLM_TEMPERATURE = 0.4
# What: reply randomness.
# Why 0.4: persona stays consistent call-to-call, phrasing still natural.
# Options: 0.7+ livelier but drifts; 0.0 robotic-consistent.

# --- TTS ---------------------------------------------------------------------

ELEVENLABS_TTS_MODEL = "eleven_flash_v2_5"
# Why: ElevenLabs' lowest-latency model (~75ms model time); measured ttfb
#   0.17-0.25s stable vs OpenAI TTS's 0.95-3.16s swings -- the single biggest
#   latency fix in this project.
ELEVENLABS_TTS_VOICE = "Xb7hH8MSUJpSbSDYk0k2"
# Swap for an Arabic-capable voice id from the ElevenLabs library when
# running PERSONA_LANG=ar.

ELEVENLABS_STREAMING_LATENCY = 3
# What: server-side quality-vs-speed dial, 0-4.
# Why 3: starts emitting audio with minimal buffering; 4 can mangle numbers.

ELEVENLABS_AUTO_MODE = True
# What: skip the input chunk schedule and synthesize as soon as a full
#   sentence arrives.
# Why True: LiveKit always sends whole sentences, so the buffering auto_mode
#   disables was pure added latency.

OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
OPENAI_TTS_VOICE = "shimmer"
# The TTS_PROVIDER=openai fallback -- kept working to demonstrate the swap.

# --- Filler ------------------------------------------------------------------

FILLER_ENABLED = True
# What: on tool calls, race a side-LLM "let me check..." line against the real
#   reply (see filler.py). Losers are never spoken.
# Why True: tool turns carry the longest silences (second LLM trip); the
#   filler makes them feel attended instead of dead.
