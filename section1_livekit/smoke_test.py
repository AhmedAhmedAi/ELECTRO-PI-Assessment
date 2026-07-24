"""Headless proof that the LLM really invokes the tools.

Runs the same Agent + real LLM as agent.py, but drives it with text instead of
audio -- no mic, no LiveKit room, no TTS spend. This is the evidence for
Task 1.1's "demonstrate the LLM actually invoking the tool call".

    .venv/bin/python smoke_test.py
"""

import asyncio
import logging
import sys

from livekit.agents import (
    AgentSession,
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
)

import config
import providers
from persona import SupportAgent

# One stream, one order: the bolt-bites TOOL CALL lines and our transcript
# prints interleave correctly instead of racing stderr against stdout.
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
# httpx/openai log every request at INFO -- too noisy to read the transcript.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

TURNS = [
    "Hi, where is my order 101?",
    "And what about order 999?",
    "Please cancel my order 202.",  # being prepared -> cancels
    "Actually cancel 101 too.",  # out for delivery -> ToolError recovery
]


async def main() -> None:
    config.validate()
    session = AgentSession(llm=providers.build_llm())
    await session.start(SupportAgent())

    for turn in TURNS:
        print(f"\n{'=' * 60}\nUSER: {turn}")
        result = await session.run(user_input=turn)

        for event in result.events:
            if isinstance(event, FunctionCallEvent):
                print(f"  -> tool call: {event.item.name}({event.item.arguments})")
            elif isinstance(event, FunctionCallOutputEvent):
                print(f"  -> tool result: {event.item.output}")
            elif isinstance(event, ChatMessageEvent):
                print(f"AGENT: {event.item.text_content}")

    await session.aclose()


if __name__ == "__main__":
    asyncio.run(main())
