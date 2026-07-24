"""The turn story, written to session.log in plain language.

Makes the invisible mechanics visible: what the agent heard, when a draft
reply was thrown away because the caller kept talking, where the agent was
cut off mid-sentence, and which overlaps were real interruptions vs noise.
Companion lines come from tools.py (TOOL CALL/RESULT/ERROR), turns.py
(PURGED) and filler.py (FILLER SAID/DISCARDED).
"""

import logging

from livekit.agents import AgentSession

logger = logging.getLogger("bolt-bites")


def attach(session: AgentSession) -> None:
    """Subscribe the narrator to a session's lifecycle events."""

    @session.on("user_input_transcribed")
    def _heard(ev) -> None:
        # Final transcript fragments. Two HEARD lines before one ASSISTANT
        # line = the caller resumed and the fragments were merged.
        if ev.is_final:
            logger.info("HEARD: %s", ev.transcript)

    @session.on("agent_state_changed")
    def _state(ev) -> None:
        # thinking -> listening without ever speaking = the draft reply (and
        # any speculative tool reads, see PURGED lines) was thrown away
        # because the caller kept talking; the agent re-decides from the
        # full, merged speech.
        if ev.old_state == "thinking" and ev.new_state == "listening":
            logger.info(
                "TURN DISCARDED: caller resumed before the agent spoke -- "
                "draft dropped, re-deciding with the complete speech"
            )

    @session.on("overlapping_speech")
    def _overlap(ev) -> None:
        if ev.is_interruption:
            logger.info(
                "BARGE-IN: caller spoke over the agent (%.2fs) -- stopping playback",
                ev.total_duration or 0,
            )
        else:
            logger.info(
                "OVERLAP ignored: backchannel while agent speaks (%.2fs)",
                ev.total_duration or 0,
            )

    @session.on("agent_false_interruption")
    def _false_interruption(ev) -> None:
        logger.info(
            "FALSE INTERRUPTION: pause was triggered but no speech followed -- %s",
            "resuming the reply" if ev.resumed else "reply stays stopped",
        )

    @session.on("conversation_item_added")
    def _item(ev) -> None:
        item = ev.item
        if item.type != "message":
            return
        if item.role == "assistant" and item.interrupted:
            # The logged text is exactly what the caller HEARD before cutting
            # in -- history is truncated to it, so the LLM's memory matches
            # the caller's ears.
            logger.info("ASSISTANT (CUT OFF HERE): %s", item.text_content)
        else:
            logger.info("%s: %s", item.role.upper(), item.text_content)
        m = item.metrics or {}
        ttft, ttfb = m.get("llm_node_ttft"), m.get("tts_node_ttfb")
        if ttft or ttfb:
            logger.info("  metrics: ttft=%.2fs ttfb=%.2fs", ttft or 0, ttfb or 0)
