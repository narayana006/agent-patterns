"""Planner-executor: plan every step up front, then run the plan with plain code.

The idea
    The model is asked once for a complete plan as JSON: a list of tool calls,
    where a later step can refer to an earlier result as "$1", "$2", ... The
    executor is ordinary code. It validates the plan, runs each step and
    substitutes the references. A final model call turns the results into an
    answer. The model never executes anything itself, and the whole plan can
    be inspected (or rejected) before the first tool runs.

Run it:  python patterns/planner_executor.py
"""

import json
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


# --- tools: plain functions the executor is allowed to call --------------------

PRICES = {"train ticket": 42.0, "hostel night": 31.5, "museum pass": 18.0}


def price_of(item: str) -> float:
    return PRICES[item.lower()]


def multiply(a: float, b: float) -> float:
    return a * b


def add(values: list[float]) -> float:
    return sum(values)


TOOLS: dict[str, Callable] = {"price_of": price_of, "multiply": multiply, "add": add}


# --- the pattern ---------------------------------------------------------------

class PlanError(ValueError):
    pass


def make_plan(goal: str, model: Model) -> list[dict]:
    prompt = (
        f"Goal: {goal}\n"
        f"Tools: {', '.join(TOOLS)}\n"
        'Reply with a JSON list of steps: {"tool": name, "args": {...}}. '
        'Use "$N" to refer to the result of step N.'
    )
    try:
        plan = json.loads(model(prompt))
    except json.JSONDecodeError as exc:
        raise PlanError(f"plan is not valid JSON: {exc.msg}") from exc
    if not isinstance(plan, list) or not plan:
        raise PlanError("plan must be a non-empty list of steps")
    for number, step in enumerate(plan, start=1):
        if not isinstance(step, dict) or step.get("tool") not in TOOLS:
            raise PlanError(f"step {number}: unknown tool {step.get('tool') if isinstance(step, dict) else step!r}")
        if not isinstance(step.get("args", {}), dict):
            raise PlanError(f"step {number}: args must be an object")
    return plan


def resolve(value, results: list):
    """Replace "$N" with the result of step N, also inside lists."""
    if isinstance(value, str) and value.startswith("$") and value[1:].isdigit():
        index = int(value[1:])
        if not 1 <= index <= len(results):
            raise PlanError(f"reference {value} points at a step that has not run")
        return results[index - 1]
    if isinstance(value, list):
        return [resolve(item, results) for item in value]
    return value


def execute(plan: list[dict]) -> list:
    results: list = []
    for number, step in enumerate(plan, start=1):
        args = {name: resolve(value, results) for name, value in step.get("args", {}).items()}
        result = TOOLS[step["tool"]](**args)
        results.append(result)
        trace(f"step {number}", f"{step['tool']}({', '.join(f'{k}={v!r}' for k, v in args.items())}) -> {result!r}")
    return results


def run(goal: str, model: Model) -> str:
    trace("goal", goal)
    plan = make_plan(goal, model)
    trace("plan", f"{len(plan)} steps: " + " -> ".join(step["tool"] for step in plan))
    results = execute(plan)
    answer = model(f"Goal: {goal}\nStep results: {json.dumps(results)}\nWrite the final answer in one sentence.")
    trace("answer", answer)
    return answer


def demo() -> str:
    plan = [
        {"tool": "price_of", "args": {"item": "train ticket"}},
        {"tool": "multiply", "args": {"a": "$1", "b": 2}},
        {"tool": "price_of", "args": {"item": "hostel night"}},
        {"tool": "multiply", "args": {"a": "$3", "b": 3}},
        {"tool": "price_of", "args": {"item": "museum pass"}},
        {"tool": "add", "args": {"values": ["$2", "$4", "$5"]}},
    ]
    model = ScriptedModel([
        json.dumps(plan),
        "The trip costs 196.50 in total: 84.00 for trains, 94.50 for the hostel and 18.00 for the museum pass.",
    ])
    answer = run("Budget a 3-night trip: return train, 3 hostel nights, one museum pass.", model)
    print(f"\nmodel calls: {len(model.prompts)} (one to plan, one to answer; the six steps ran as plain code)")
    return answer


if __name__ == "__main__":
    demo()
