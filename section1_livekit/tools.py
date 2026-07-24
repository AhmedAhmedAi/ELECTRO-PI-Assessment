"""What Sara can do: the tools the LLM may call.

Standalone functions, not Agent methods -- persona.py hands them to the agent.
The signature + docstring of each function IS the schema the LLM sees.
"""

import logging

from livekit.agents import RunContext, function_tool
from livekit.agents.llm import ToolError

import config
import filler
import orders

logger = logging.getLogger("bolt-bites")


@function_tool()
async def get_order_status(context: RunContext, order_id: str) -> str:
    """Look up the delivery status of a Bolt Bites order.

    Args:
        order_id: The order ID, e.g. "101".
    """
    await filler.fire_and_delay(context, config.TOOL_LATENCY_MS)
    logger.info("TOOL CALL  get_order_status(order_id=%r)", order_id)

    result = orders.order_status(order_id)
    logger.info("TOOL RESULT %s", result)
    return result


@function_tool()
async def cancel_order(context: RunContext, order_id: str) -> str:
    """Cancel a Bolt Bites order for the caller.

    Args:
        order_id: The order ID to cancel, e.g. "202".
    """
    await filler.fire_and_delay(context, config.CANCEL_LATENCY_MS)
    logger.info("TOOL CALL  cancel_order(order_id=%r)", order_id)
    try:
        result = orders.cancel_order(order_id)
    except (LookupError, ValueError) as e:
        # The ToolError text goes back to the LLM, so it explains the refusal
        # out loud instead of the session crashing.
        logger.info("TOOL ERROR %s", e)
        raise ToolError(str(e)) from e

    logger.info("TOOL RESULT %s", result)
    return result
