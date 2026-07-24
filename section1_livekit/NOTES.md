# Write-up: barge-in / interruption handling, and adding a second tool safely

Barge-in really happens in two different situations, and they need
different handling. These are techniques I use in my own products.

**Situation 1 — the user interrupts before the agent speaks** (the LLM is
still generating, or finished but nothing was voiced yet). I abort the
in-flight LLM request — or if it already completed, I ignore the response
entirely, as if it was never submitted — and the user's new words get
appended to their previous utterance and re-sent as one completed request.
This is exactly why I run the turn-end delay at zero: it doesn't matter if
a request fires too early, because I can always cancel it and resubmit the
completed sentence. It's a deliberate trade — I pay for some wasted LLM
calls to get close to the feel of a live voice model, at a much lower
cost. One guard belongs here: check the turn's tool log before discarding
— if a tool already ran, the turn can't be thrown away silently.

**Situation 2 — the user interrupts while the agent is speaking.** Here I
don't delete, I flag: mark the response as interrupted, mark where the TTS
stopped, and mark any tool that was called. The flags live in the chat
history so the model always knows what the user actually heard versus what
it only generated. I also can't discard the unspoken text — live agents
usually show the reply as text while speaking it, and the user may want to
go back to it.

For the full live-model experience I would go multimodal — a model that
takes audio directly (Gemini Flash is my pick for these apps): submit the
moment the user stops, terminate if they keep talking. My reference for
building this without a live model is Claude's voice mode.

**Adding a second tool safely:** the schema comes from typed parameters
plus the docstring — the docstring is the tool's prompt. Every failure
becomes a `ToolError` with a speakable reason, so the model recovers out
loud ("the courier is already on the way") instead of dead air, and
external calls get timeouts. I keep a strict read/write split: read tools
are replayable and safe to purge after an aborted turn; write tools are
never purged — their effects already happened — and a destructive one
gets a spoken confirmation first.
