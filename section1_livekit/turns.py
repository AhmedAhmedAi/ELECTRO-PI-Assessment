"""Turn-history hygiene: purge speculative reads left by aborted turns.

The motivating bug (logged in an earlier build with longer IDs like
BB-1042): a caller said "It's BB-10" [pause] "42". The pause committed the
turn, the LLM looked up BB-10 -> not found, then "42" arrived before the
agent spoke. LiveKit discarded the un-spoken draft and merged the
fragments, but the lookup's function_call/output stayed in history -- so
the regenerated reply said "couldn't find it" instead of re-searching.

Rule: tool READS are speculative, safe to drop and re-run. Tool WRITES
changed reality (a cancel that ran, ran) -- their artifacts always stay.
"""

import logging

logger = logging.getLogger("bolt-bites")

READ_ONLY_TOOLS = {"get_order_status"}


def find_stale_readonly_tool_ids(items: list) -> list[str]:
    """Ids of read-only tool artifacts left by an aborted (never-spoken) turn.

    A completed turn always commits its spoken assistant message LAST; an
    aborted one never does. So only the tail after the last assistant message
    can hold stale items. User messages in that tail are the merged speech
    fragments -- kept. Pure function, unit-testable without a session.
    """
    last_assistant = -1
    for i, it in enumerate(items):
        if it.type == "message" and it.role == "assistant":
            last_assistant = i
    tail = items[last_assistant + 1 :]

    # Outputs don't reliably carry the tool name -- match read-vs-write via
    # the call_id of their function_call.
    readonly_call_ids = {
        it.call_id
        for it in tail
        if it.type == "function_call" and it.name in READ_ONLY_TOOLS
    }

    remove_ids: list[str] = []
    for it in tail:
        if it.type == "function_call" and it.name in READ_ONLY_TOOLS:
            remove_ids.append(it.id)
        elif it.type == "function_call_output" and it.call_id in readonly_call_ids:
            remove_ids.append(it.id)
    return remove_ids


async def purge_stale_reads(agent, turn_ctx) -> None:
    """Drop aborted-turn read artifacts and make the removal stick."""
    remove_ids = find_stale_readonly_tool_ids(turn_ctx.items)
    if not remove_ids:
        return

    by_id = {it.id: it for it in turn_ctx.items}
    for rid in remove_ids:
        it = by_id[rid]
        args = getattr(it, "arguments", None) or getattr(it, "call_id", "")
        name = getattr(it, "name", "") or it.type
        logger.info("PURGED stale %s(%s) from aborted turn", name, args)

    # Cleans the context the upcoming reply generates from...
    turn_ctx.items[:] = [it for it in turn_ctx.items if it.id not in remove_ids]
    # ...and persists it: turn_ctx is a throwaway copy (the SDK builds it as
    # chat_ctx.copy()), so without this write-back the stale items would
    # reappear next turn.
    await agent.update_chat_ctx(turn_ctx)
