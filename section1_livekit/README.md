# Section 1 — "Sara", a real-time voice support agent

I built Sara, a phone-support voice agent for a food-delivery app, on the
livekit-agents pipeline: STT → LLM → TTS, with two tools the LLM calls
mid-conversation against a mock order store. She answers in ~1.3 seconds,
speaks English or Egyptian Arabic, survives a provider dying mid-call, and
handles interruptions without corrupting her own memory. Evidence of all
of it: [transcript.md](transcript.md), plus an 8-minute
[video of a live session](https://drive.google.com/file/d/1qBo52Xbigp-0x06KQy6bGPYHbL8cvdy8/view?usp=sharing)
(mic + speakers + the live webview on screen). The spec write-up (barge-in +
second tool) is in [NOTES.md](NOTES.md).

## Architecture

One pipeline, every stage streaming, every vendor swappable:

```
mic → Silero VAD (local) → OpenAI STT (streaming)
    → LLM  Gemini 3.1 Flash Lite   → fallback: OpenAI
    → TTS  ElevenLabs Flash        → fallback: OpenAI
    → speakers
```

`agent.py` wires it; `providers.py` builds every stage from config, so a
vendor swap is an env var (that's Task 1.2) — and beyond swapping, the LLM
and TTS run behind runtime fallback chains (`FallbackAdapter`), so a vendor
outage degrades the call instead of killing it. It has happened live: the
transcript shows a mid-call failover.

The persona is bilingual by config (`PERSONA_LANG=ar` for Egyptian Arabic —
instructions, greeting, and voice all switch together), and the STT gets
vocabulary hints so spoken order numbers come back as digits.

## The techniques I want you to see

**1. Turn-taking tuned hot, made safe.** I run the timers aggressively —
VAD 0.35 s, turn-end delay zero — so Sara starts thinking the moment you
stop talking. The known cost is turns that commit too early, and I made
that safe instead of slow: if you resume before she speaks, the fragments
merge and the draft reply is discarded. This is a deliberate trade — I pay
for some wasted LLM calls to get close to a live-model feel at a fraction
of the cost.

**2. The aborted-turn purge (`turns.py`).** The bug that motivated it: a
caller said an order ID, paused mid-number, and the turn committed early —
the LLM looked up a half ID, found nothing, and that failed lookup stayed
in history poisoning the retry. My rule: tool *reads* are speculative, so
the purge strips them from aborted turns and the model re-decides from
clean history; tool *writes* changed reality — a cancel that ran, ran —
so their artifacts are never purged.

**3. The filler race (`filler.py`).** On every tool call, a small side-LLM
writes a contextual "let me check that for you…" line and races the real
answer. It only gets spoken if it's ready before the tool returns —
otherwise it's discarded and never enters history. Dead air covered, zero
latency added. Try it: `TOOL_LATENCY_MS=800` and you'll hear it win.

**4. Speed with receipts (`tuning.py`).** The all-OpenAI baseline measured
~3–4 s to first sound. I got it to ~1.2–1.5 s and documented every step:

| Stage | Time |
|---|---|
| VAD end-of-speech | 0.35 s |
| Streaming STT final | ~0.15 s |
| LLM first token (Gemini flash-lite, thinking pinned to minimal) | ~0.40 s |
| TTS first byte (ElevenLabs flash) | ~0.17 s |
| **Silence → her first sound** | **~1.2–1.5 s** |

The sneaky one: Gemini 3 is a reasoning model, and unpinned thinking
silently adds seconds per reply.

**5. Tools that fail out loud (`tools.py` + `orders.py`).** LiveKit builds
each tool schema from the Python signature + docstring, so the docstring is
written for the model. The store *raises* on refusals (unknown ID, courier
already dispatched, delivered) and the tool converts each into
`ToolError(reason)` — Sara explains the failure and offers an alternative
("the courier is already on the way — want me to contact them?") instead
of dead air. The read tool participates in the purge; `cancel_order`
never does, and has its own latency knob.

**6. A frontend that shows what the caller actually experiences.** While
the console runs, http://localhost:8899 shows the call in real time: your
words appearing as they're recognized, her reply typing in sync with her
voice, and chips marking barge-ins, merged turns, and tool calls. This is
the same text-while-speaking experience my write-up's interruption
handling protects — what she said out loud stays on screen, in sync.
The page opens with the **mic muted** (like joining a call — frame your
recording, then click the button to go live; console-only runs keep a hot
mic). Bubbles follow *spoken* order: resuming before she answers keeps
growing your bubble, draft text she never actually delivered is replaced
in place by the reply she did speak, and a barge-in always starts fresh
bubbles below her cut-off one.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Put **three keys** in `.env` (the defaults use all three vendors):

| Key | Used for |
|---|---|
| `OPENAI_API_KEY` | STT (always) + fallback LLM/TTS |
| `GOOGLE_API_KEY` | primary LLM (Gemini 3.1 Flash Lite) |
| `ELEVENLABS_API_KEY` | primary TTS |

Only have OpenAI? Set `LLM_PROVIDER=openai` and `TTS_PROVIDER=openai` — the
whole pipeline runs on one key.

```bash
.venv/bin/python agent.py console    # voice: mic + speakers (use headphones)
.venv/bin/python smoke_test.py       # headless: text in, tool calls out

TOOL_LATENCY_MS=800 .venv/bin/python agent.py console    # slow "backend" → hear the filler race win
CANCEL_LATENCY_MS=1500 ...                               # slow cancels only
```

## What's inside

| File | Role |
|---|---|
| `agent.py` | wire the pipeline, run it |
| `persona.py` | who Sara is (EN/AR instructions, greeting, STT hints) |
| `tools.py` | what she can do (`get_order_status`, `cancel_order`) |
| `orders.py` | mock backend (the seam for a real one) |
| `turns.py` | purge speculative reads from aborted turns |
| `filler.py` | contextual "let me check…" raced against the real reply |
| `providers.py` | vendor factories + runtime fallback chains |
| `tuning.py` | every measured decision, with its rationale |
| `config.py` | keys and switches from `.env` |
| `logs.py` / `webview.py` + `webview.html` | narrated `session.log` / live browser view |

## Honest limitations

- Sub-1s latency needs provider co-location (measured from Cairo to US/EU
  endpoints); from a laptop, ~1.2 s is the floor — it's network, not code.
- The mock in-memory store stands in for a real backend; the seam is
  `orders.py`.
- STT has one vendor (OpenAI) — full independence needs a third key.
- TTS failover audibly changes the voice mid-call (accepted trade-off).
- Aggressive turn-taking can fragment dictated numbers; the purge makes
  that self-correcting rather than wrong.
- Harmless: transient "silero inference slower than realtime" under CPU
  load; the framework's `aclose` RuntimeWarning at exit (upstream).
- `session.log` is rewritten each run; keep a copy to preserve a session.
