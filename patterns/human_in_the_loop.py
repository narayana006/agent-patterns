"""Human-in-the-loop: the agent proposes, a person approves anything risky.

The idea
    The model proposes one tool call at a time as JSON. Tools are tagged safe
    or risky. Safe tools run straight away. A risky tool is paused and shown
    to an approver, who can approve it or deny it with a reason. A denial is
    not an error: it goes back to the model as an observation so it can
    choose another way to finish the job. The approver is just a function,
    so it can be a terminal prompt, a chat button or a ticket queue.

Run it:  python patterns/human_in_the_loop.py                (scripted approver)
         python patterns/human_in_the_loop.py --interactive  (you approve)
"""

import json
import sys
from dataclasses import dataclass
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


# --- tools, each marked safe or risky -------------------------------------------

LEDGER: list[str] = []  # stands in for the outside world


def look_up_order(order_id: str) -> str:
    return f"order {order_id}: 1 x desk lamp, 39.00, delivered damaged"


def issue_refund(order_id: str, amount: float) -> str:
    LEDGER.append(f"refund {amount:.2f} on {order_id}")
    return f"refunded {amount:.2f} on order {order_id}"


def send_voucher(order_id: str, amount: float) -> str:
    LEDGER.append(f"voucher {amount:.2f} for {order_id}")
    return f"voucher for {amount:.2f} sent for order {order_id}"


TOOLS = {  # name: (function, needs approval?)
    "look_up_order": (look_up_order, False),
    "issue_refund": (issue_refund, True),
    "send_voucher": (send_voucher, True),
}


@dataclass
class Decision:
    approved: bool
    reason: str = ""


Approver = Callable[[str, dict], Decision]


def ask_in_terminal(tool: str, args: dict) -> Decision:
    answer = input(f"Approve {tool}({args})? [y/N] ").strip().lower()
    if answer in ("y", "yes"):
        return Decision(True)
    return Decision(False, input("Reason for the agent: ").strip() or "denied by reviewer")


# --- the pattern ---------------------------------------------------------------

def run(task: str, model: Model, approve: Approver, max_steps: int = 6) -> str:
    history = [f"Task: {task}"]
    trace("task", task)
    for _ in range(max_steps):
        prompt = "\n".join(history) + (
            f"\nTools: {', '.join(TOOLS)}. "
            'Reply with JSON: {"tool": name, "args": {...}} or {"final": "message to the customer"}.'
        )
        action = json.loads(model(prompt))
        if "final" in action:
            trace("final", action["final"])
            return action["final"]
        name, args = action["tool"], action.get("args", {})
        if name not in TOOLS:
            observation = f"error: no tool called {name}"
        else:
            function, risky = TOOLS[name]
            trace("propose", f"{name}({args})" + ("  <- needs approval" if risky else ""))
            decision = approve(name, args) if risky else Decision(True)
            if decision.approved:
                if risky:
                    trace("human", "approved")
                observation = function(**args)
            else:
                trace("human", f"denied: {decision.reason}")
                observation = f"{name} was denied by a human reviewer: {decision.reason}"
        trace("observe", observation)
        history.append(f"Action: {json.dumps(action)}\nObservation: {observation}")
    raise RuntimeError(f"no final answer after {max_steps} steps")


def demo(approve: Approver | None = None) -> str:
    LEDGER.clear()
    model = ScriptedModel([
        '{"tool": "look_up_order", "args": {"order_id": "A-1042"}}',
        '{"tool": "issue_refund", "args": {"order_id": "A-1042", "amount": 390.0}}',
        '{"tool": "issue_refund", "args": {"order_id": "A-1042", "amount": 39.0}}',
        '{"final": "Sorry about the damaged lamp. I have refunded 39.00 to your original payment method."}',
    ])
    if approve is None:
        scripted = [Decision(False, "amount is 10x the order value; the order total is 39.00"), Decision(True)]
        approve = lambda tool, args: scripted.pop(0)  # noqa: E731
    try:
        final = run("Customer says the lamp in order A-1042 arrived broken. Make it right.", model, approve)
    except RuntimeError as exc:  # the canned replies only fit one approve/deny path
        final = f"stopped: {exc}"
        trace("stopped", str(exc))
    print(f"\nside effects: {LEDGER if LEDGER else 'none'}")
    return final


if __name__ == "__main__":
    if "--interactive" in sys.argv:
        print("The model's replies are scripted, so its next proposal will not react to your choice.")
        print("Watch what does and does not run: deny everything and the ledger stays empty.\n")
        demo(ask_in_terminal)
    else:
        demo()
