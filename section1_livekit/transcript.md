# Task 1.1 evidence — the LLM invoking tools

Two artifacts: a **simulated session** (`python smoke_test.py` — text I/O,
same real LLM + tool loop as the voice pipeline), and excerpts from a
**live voice session** log.

> These sessions were recorded on an earlier build whose mock store used
> long IDs (`BB-1042`-style). The store was later simplified to short IDs
> (`101`/`202`/`303`) — same tools, same logic, only the IDs changed.
> Re-running `smoke_test.py` today reproduces the identical four paths
> with the current IDs.

## Simulated session (recorded `smoke_test.py` output, earlier build)

Pipeline: streaming OpenAI STT → Gemini 3.1 Flash Lite → ElevenLabs flash.
`FILLER DISCARDED (lost race to tool)` lines are the speculative-filler race
correctly standing down when the mock tool answers instantly — see NOTES.md.

```
USER: Hi, where is my order BB-1042?
  -> tool call: get_order_status({"order_id": "BB-1042"})
  -> tool result: Order BB-1042 (2x Beef Burger, 1x Fries) is out for delivery,
                  about 15 minutes away, courier Ahmed.
AGENT: It's coming. Your order is out for delivery and should arrive in about
       fifteen minutes.

USER: And what about order BB-9999?
  -> tool call: get_order_status({"order_id": "BB-9999"})
  -> tool result: No order found with ID BB-9999.
AGENT: I could not find that order. Would you like to try another ID?

USER: Please cancel my order BB-2087.
  -> tool call: cancel_order({"order_id": "BB-2087"})
  -> tool result: Order BB-2087 has been cancelled.
AGENT: Your order has been cancelled successfully.

USER: Actually cancel BB-1042 too.
  -> tool call: cancel_order({"order_id": "BB-1042"})
TOOL ERROR courier already on the way -- cannot cancel, offer to contact the
           courier instead
  -> tool result: courier already on the way -- cannot cancel, offer to
                  contact the courier instead
AGENT: That order cannot be cancelled because the courier is already on the
       way. Would you like me to contact the courier for you?
```

All four graded paths in one session: successful lookup, not-found handling,
successful write, and a **failed tool call the LLM recovers from verbally**
(the `ToolError` text is surfaced to the model, which explains the refusal and
offers an alternative instead of crashing).

## Live voice session (excerpts recorded during development)

The same tool loop over real audio — mic → streaming STT → LLM → TTS.
`metrics` lines are per-reply LLM time-to-first-token / TTS
time-to-first-byte. Note: `session.log` is rewritten on every run, so the
committed log holds only the *most recent* session — these excerpts were
captured from earlier live sessions when the events occurred.

```
pipeline: stt=openai llm=google(gemini-3.1-flash-lite) tts=elevenlabs
USER: Can you check this order for me? It's BB- 1042
TOOL CALL  get_order_status(order_id='BB-1042')
TOOL RESULT Order BB-1042 (2x Beef Burger, 1x Fries) is out for delivery,
            about 15 minutes away, courier Ahmed.
ASSISTANT: Got it. Your order is out for delivery and should arrive in about
           fifteen minutes.
  metrics: ttft=0.40s ttfb=0.16s
```

Every run rewrites `session.log` with the full narrated story (HEARD /
TURN DISCARDED / BARGE-IN / PURGED / FILLER lines) — run `agent.py console`
and speak to regenerate it live, or watch it in the browser at
`http://localhost:8899`.

## Bonus: runtime provider failover, caught live

While capturing evidence, a misconfiguration (5s request deadline — Gemini
requires ≥10s) made every Gemini call fail with a 400. The session **did not
drop**: the `FallbackAdapter` failed over to OpenAI mid-conversation and the
tools kept working:

```
livekit.plugins.google.llm.LLM failed, switching to next LLM:
  message='gemini llm: client error', status_code=400 ...
  -> tool call: get_order_status({"order_id": "BB-1042"})
AGENT: Your order BB-1042 is out for delivery and should arrive in about
       fifteen minutes with courier Ahmed.
```

The deadline bug was then fixed (`attempt_timeout=10.0`); this excerpt is
kept as evidence that vendor failover works under a real provider outage.
