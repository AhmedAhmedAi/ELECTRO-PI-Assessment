"""Speculative "smart filler": a spoken bridge while a tool runs.

When the LLM calls a tool, the reply goes quiet (tool + a second LLM trip).
A parallel side-LLM generates a short contextual line ("Let me pull up that
order") and races the real reply:

  - filler ready BEFORE the tool returns -> spoken first, reply follows
  - tool returned first                  -> discarded, never spoken

Losers are dropped silently; a filler failure must never break the call.
"""

import asyncio
import logging

from livekit.agents import llm as llmmod

import config
import tuning

logger = logging.getLogger("bolt-bites")

_SYSTEM_PROMPT = (
    "You are Sara, a phone support agent. Produce ONE short natural spoken "
    "filler sentence, max 8 words, telling the caller you are checking -- in "
    "the same language as the caller. Plain text, no markdown."
)
# The last messages usually reveal the language, but the caller may have said
# only an order ID -- pin it explicitly in Arabic mode.
_LANG_HINTS = {"ar": " The caller speaks Egyptian Arabic; answer in it."}


async def fire_and_delay(context, latency_ms: int) -> None:
    """Tool entry point: start the race, then simulate backend latency.

    Never awaits the race -- that would serialize the exact gap the filler
    exists to cover. tool_done marks the race deadline: once the tool returns,
    the framework queues the real reply, and anything said after that would
    play BEHIND the answer (backwards). So the filler must beat the tool.
    """
    tool_done = asyncio.Event()
    if tuning.FILLER_ENABLED:
        asyncio.create_task(race_filler(context.session, tool_done))
    if latency_ms:
        await asyncio.sleep(latency_ms / 1000)
    tool_done.set()


async def race_filler(session, tool_done: asyncio.Event) -> None:
    """Generate a filler; speak it only if it beat the tool's return."""
    try:
        # Last 2 chat items give the side-LLM just enough context to phrase
        # a relevant bridge.
        text = await generate_filler(
            session.llm, session.history.items[-2:], config.PERSONA_LANG
        )
        if not text:
            return

        # No await between these checks and say(): single event loop, so the
        # tool cannot slip in between.
        if tool_done.is_set():
            logger.info("FILLER DISCARDED (lost race to tool): %s", text)
            return
        if session.agent_state == "speaking":
            logger.info("FILLER DISCARDED (reply already speaking): %s", text)
            return
        if session.output.audio is None:
            logger.info("FILLER DISCARDED (no audio output): %s", text)
            return

        # add_to_chat_ctx=False: the filler is throwaway speech -- recording it
        # would also break turns.py's aborted-turn detection, which keys off
        # the last assistant message.
        session.say(text, add_to_chat_ctx=False)
        logger.info("FILLER SAID: %s", text)
    except Exception:
        logger.exception("FILLER DISCARDED (error, ignored)")


async def generate_filler(llm_instance, recent_items, lang: str) -> str:
    """One filler line from the last couple of chat messages. "" on failure."""
    chat_ctx = llmmod.ChatContext.empty()
    chat_ctx.add_message(role="system", content=_SYSTEM_PROMPT + _LANG_HINTS.get(lang, ""))

    messages = [it for it in recent_items if getattr(it, "type", None) == "message"]
    for it in messages[-2:]:
        text = (it.text_content or "").strip()
        if text:
            role = "assistant" if it.role == "assistant" else "user"
            chat_ctx.add_message(role=role, content=text)

    # to_str_iterable() opens and closes the LLM stream for us.
    parts: list[str] = []
    async for delta in llm_instance.chat(chat_ctx=chat_ctx).to_str_iterable():
        parts.append(delta)
    return "".join(parts).strip()
