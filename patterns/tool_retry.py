"""Tool retry and error recovery: retry what might work, report what will not.

The idea
    Tool failures come in two kinds. Transient ones (timeouts, rate limits)
    are retried by code with exponential backoff; the model never hears
    about them unless the retries run out. Permanent ones (bad arguments,
    unknown records) are not retried, because the same call would fail
    again. Instead the error text is returned to the model as an
    observation, so it can correct its arguments or switch to a fallback.

Run it:  python patterns/tool_retry.py
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


class TransientError(Exception):
    """Worth retrying: the same call may succeed in a moment."""


class PermanentError(Exception):
    """Not worth retrying: the call itself is wrong."""


# --- tools: a flaky live service and a dependable but stale fallback ------------

class FlakyWeatherService:
    """Fails with a timeout a set number of times, then answers."""

    def __init__(self, failures_per_city: dict[str, int]):
        self.remaining = dict(failures_per_city)
        self.calls = 0

    def __call__(self, city: str) -> str:
        self.calls += 1
        known = {"Lisbon": "18C, clear", "Oslo": "3C, snow"}
        if city not in known:
            raise PermanentError(f"unknown city {city!r}; known cities: {', '.join(known)}")
        if self.remaining.get(city, 0) > 0:
            self.remaining[city] -= 1
            raise TransientError("upstream timeout")
        return known[city]


def cached_weather(city: str) -> str:
    return {"Lisbon": "17C, clear (cached 2h ago)", "Oslo": "2C, cloudy (cached 2h ago)"}[city]


# --- the pattern ---------------------------------------------------------------

def call_with_retry(tool: Callable, args: dict, attempts: int = 3, base_delay: float = 0.5,
                    sleep: Callable[[float], None] = lambda seconds: None) -> str:
    """Run a tool, retrying transient failures. Returns an observation string either way.

    ``sleep`` is injectable so the demo and the tests do not actually wait;
    pass ``time.sleep`` in real use.
    """
    for attempt in range(1, attempts + 1):
        try:
            return f"ok: {tool(**args)}"
        except TransientError as exc:
            if attempt == attempts:
                return f"error: still failing after {attempts} attempts ({exc}); try another tool"
            delay = base_delay * 2 ** (attempt - 1)
            trace("retry", f"attempt {attempt} failed ({exc}); waiting {delay:.1f}s")
            sleep(delay)
        except PermanentError as exc:
            return f"error: {exc}"  # do not retry; let the model fix the call
        except Exception as exc:  # anything unexpected is treated as permanent
            return f"error: {type(exc).__name__}: {exc}"
    raise AssertionError("unreachable")


def run(question: str, model: Model, tools: dict[str, Callable], max_steps: int = 6,
        sleep: Callable[[float], None] = lambda seconds: None) -> str:
    history = [f"Question: {question}"]
    trace("question", question)
    for _ in range(max_steps):
        prompt = "\n".join(history) + (
            f"\nTools: {', '.join(tools)}. "
            'Reply with JSON: {"tool": name, "args": {...}} or {"final": "answer"}.'
        )
        try:
            action = json.loads(model(prompt))
        except json.JSONDecodeError:
            observation = "error: reply was not valid JSON"
            action = {}
        else:
            if "final" in action:
                trace("final", action["final"])
                return action["final"]
            name, args = action.get("tool"), action.get("args", {})
            trace("call", f"{name}({args})")
            if name not in tools:
                observation = f"error: no tool called {name!r}"
            else:
                observation = call_with_retry(tools[name], args, sleep=sleep)
        trace("observe", observation)
        history.append(f"Action: {json.dumps(action)}\nObservation: {observation}")
    return "Sorry, I could not complete that."


def demo() -> str:
    live = FlakyWeatherService({"Lisbon": 2, "Oslo": 99})
    waits: list[float] = []
    model = ScriptedModel([
        '{"tool": "live_weather", "args": {"city": "Lisbon"}}',      # 2 timeouts, then fine
        '{"tool": "live_weather", "args": {"city": "Olso"}}',        # typo: permanent error
        '{"tool": "live_weather", "args": {"city": "Oslo"}}',        # corrected, but service is down
        '{"tool": "cached_weather", "args": {"city": "Oslo"}}',      # fallback
        '{"final": "Lisbon is 18C and clear. Oslo was 2C and cloudy two hours ago; live data is unavailable."}',
    ])
    answer = run("What is the weather in Lisbon and Oslo?", model,
                 {"live_weather": live, "cached_weather": cached_weather}, sleep=waits.append)
    print(f"\nlive service calls: {live.calls}; model calls: {len(model.prompts)}; "
          f"backoff waits: {waits} (not actually slept in the demo)")
    return answer


if __name__ == "__main__":
    demo()
