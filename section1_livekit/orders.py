"""Mock order store.

Stands in for whatever the real backend would be (Postgres, an internal REST
service). Isolated here so persona.py stays pure agent logic and the seam for
a real integration is obvious.
"""

# Short numeric IDs on purpose: easy to say, and STT transcribes them
# reliably — no ID-normalization layer needed.
_ORDERS = {
    "101": {
        "status": "out for delivery",
        "eta_minutes": 15,
        "items": "2x Beef Burger, 1x Fries",
        "courier": "Ahmed",
    },
    "202": {
        "status": "being prepared",
        "eta_minutes": 35,
        "items": "1x Margherita Pizza",
        "courier": None,
    },
    "303": {
        "status": "delivered",
        "eta_minutes": 0,
        "items": "3x Chicken Shawarma, 2x Pepsi",
        "courier": "Mariam",
    },
}


def order_status(order_id: str) -> str:
    """One-sentence status for the LLM to relay -- the read twin of
    cancel_order, so both tools speak through this module."""
    key = order_id.strip()
    order = _ORDERS.get(key)
    if order is None:
        return f"No order found with ID {order_id}."
    if order["status"] == "delivered":
        return f"Order {key} ({order['items']}) was delivered by {order['courier']}."
    if order["courier"]:
        return (
            f"Order {key} ({order['items']}) is {order['status']}, "
            f"about {order['eta_minutes']} minutes away, courier {order['courier']}."
        )
    return (
        f"Order {key} ({order['items']}) is {order['status']}, "
        f"ready in about {order['eta_minutes']} minutes."
    )


def cancel_order(order_id: str) -> str:
    """Cancel an order, or refuse when its state does not allow it.

    Raises so persona.py can turn each refusal into a spoken explanation:
    LookupError for a missing ID, ValueError for a valid order too far along
    to cancel.
    """
    key = order_id.strip()
    order = _ORDERS.get(key)

    if order is None:
        raise LookupError(f"No order found with ID {order_id}.")

    status = order["status"]
    if status == "cancelled":
        return f"Order {key} is already cancelled."
    if status == "being prepared":
        order["status"] = "cancelled"
        return f"Order {key} has been cancelled."
    if status == "out for delivery":
        raise ValueError(
            "courier already on the way -- cannot cancel, offer to contact the courier instead"
        )
    if status == "delivered":
        raise ValueError("already delivered -- suggest a refund request")

    raise ValueError(f"Order {key} cannot be cancelled from status {status}.")
