"""Router: classify the input once, then dispatch to a specialist handler.

The idea
    One cheap model call picks a route from a fixed list. Each route has its
    own handler with its own prompt (or no model at all). The router never
    trusts the label blindly: anything outside the list goes to a fallback.

Run it:  python patterns/router.py
"""

from typing import Callable

Model = Callable[[str], str]  # prompt in, text out: wrap any provider in this


class ScriptedModel:
    """Offline stand-in for a real model: replays canned replies in order."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise RuntimeError("ScriptedModel ran out of replies")
        return self.replies.pop(0)


def trace(step: str, text: str) -> None:
    print(f"[{step:<8}] {text}")


# --- handlers: one per route --------------------------------------------------

def handle_billing(message: str, model: Model) -> str:
    return model(f"You are a billing specialist. Answer briefly.\n\nCustomer: {message}")


def handle_technical(message: str, model: Model) -> str:
    return model(f"You are a support engineer. Give numbered steps.\n\nCustomer: {message}")


def handle_general(message: str, model: Model) -> str:
    # Not every route needs a model. A canned reply is cheaper and predictable.
    return "Thanks for getting in touch. A member of the team will reply within one working day."


ROUTES: dict[str, Callable[[str, Model], str]] = {
    "billing": handle_billing,
    "technical": handle_technical,
    "general": handle_general,
}
FALLBACK = "general"


# --- the pattern ---------------------------------------------------------------

def classify(message: str, model: Model) -> str:
    """Ask for exactly one label and normalise whatever comes back."""
    prompt = (
        f"Classify the customer message into one of: {', '.join(ROUTES)}.\n"
        "Reply with the label only.\n\n"
        f"Message: {message}"
    )
    raw = model(prompt)
    label = raw.strip().strip(".").lower()
    if label not in ROUTES:
        trace("router", f"unknown label {raw!r}, falling back to {FALLBACK!r}")
        return FALLBACK
    return label


def route(message: str, model: Model) -> tuple[str, str]:
    """Return (route, reply) for one message."""
    trace("input", message)
    label = classify(message, model)
    trace("router", f"route = {label}")
    reply = ROUTES[label](message, model)
    trace("reply", reply)
    return label, reply


def demo() -> list[tuple[str, str]]:
    model = ScriptedModel([
        "billing",
        "You were charged twice because the first payment was retried. The duplicate will be refunded in 3 to 5 days.",
        "Technical.",
        "1. Open Settings > Sync. 2. Tap 'Reset sync state'. 3. Sign in again.",
        "I think this is probably a complaint?",  # not a valid label
    ])
    messages = [
        "I was charged twice for my subscription this month.",
        "The app stopped syncing my notes after the last update.",
        "Do you have an office in Lisbon?",
    ]
    results = []
    for message in messages:
        results.append(route(message, model))
        print()
    print(f"model calls: {len(model.prompts)} (3 to classify, 2 to answer; 'general' needs none)")
    return results


if __name__ == "__main__":
    demo()
