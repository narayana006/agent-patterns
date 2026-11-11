"""Evaluator-optimizer: draft, critique, revise, and stop when it passes.

The idea
    One call writes a draft. An evaluator scores it against explicit criteria
    and returns feedback. The feedback goes into the next drafting prompt.
    The loop ends when the evaluator passes the draft or the round limit is
    hit. Checks that code can do (length, required words) are done in code;
    the model is only asked for the judgement code cannot make.

Run it:  python patterns/evaluator_optimizer.py
"""

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


@dataclass
class Verdict:
    passed: bool
    feedback: str


# --- the pattern ---------------------------------------------------------------

def generate(task: str, model: Model, previous: str | None, feedback: str | None) -> str:
    prompt = f"Task: {task}"
    if previous is not None:
        prompt += f"\n\nYour previous draft:\n{previous}\n\nFix these problems:\n{feedback}"
    return model(prompt).strip()


def evaluate(task: str, draft: str, model: Model, max_words: int, must_mention: list[str]) -> Verdict:
    # 1. Rules first: free, instant and never wrong about what they measure.
    problems = []
    words = len(draft.split())
    if words > max_words:
        problems.append(f"Too long: {words} words, the limit is {max_words}.")
    for term in must_mention:
        if term.lower() not in draft.lower():
            problems.append(f"Does not mention '{term}'.")
    if problems:
        return Verdict(False, " ".join(problems))
    # 2. Only then spend a model call on the part that needs judgement.
    reply = model(
        f"Task: {task}\n\nDraft:\n{draft}\n\n"
        "Is the draft clear and free of jargon? Reply 'PASS' or 'REVISE: <what to fix>'."
    ).strip()
    if reply.upper().startswith("PASS"):
        return Verdict(True, "")
    return Verdict(False, reply.removeprefix("REVISE:").strip())


def optimize(task: str, writer: Model, critic: Model, max_rounds: int = 4,
             max_words: int = 30, must_mention: tuple[str, ...] = ()) -> tuple[str, bool, int]:
    """Return (best draft, whether it passed, rounds used)."""
    draft, feedback = None, None
    for round_number in range(1, max_rounds + 1):
        draft = generate(task, writer, draft, feedback)
        trace(f"draft {round_number}", draft)
        verdict = evaluate(task, draft, critic, max_words, list(must_mention))
        if verdict.passed:
            trace("evaluate", "PASS")
            return draft, True, round_number
        feedback = verdict.feedback
        trace("evaluate", f"REVISE: {feedback}")
    trace("stop", f"round limit ({max_rounds}) reached; returning the last draft unapproved")
    return draft, False, max_rounds


def demo() -> tuple[str, bool, int]:
    task = "Write a status-page update about the login outage for customers."
    writer = ScriptedModel([
        "We are currently experiencing an incident impacting the authentication subsystem whereby a subset of "
        "requests to the token issuance endpoint are returning elevated error rates, and our engineers are "
        "actively investigating the root cause of the degradation.",
        "Some sign-ins are failing because of a fault in our auth token cache. A fix is being rolled out.",
        "Some customers cannot sign in right now. We have found the cause and expect a fix within 30 minutes.",
    ])
    critic = ScriptedModel([
        # Round 1 never reaches the critic: the rule checks fail first.
        "REVISE: 'auth token cache' is jargon. Say what customers see, and when it will be fixed.",
        "PASS",
    ])
    final, passed, rounds = optimize(task, writer, critic, max_words=30, must_mention=("sign",))
    print(f"\nresult: {'approved' if passed else 'not approved'} after {rounds} rounds")
    print(f"model calls: {len(writer.prompts)} drafts, {len(critic.prompts)} critiques (round 1 failed on rules alone)")
    return final, passed, rounds


if __name__ == "__main__":
    demo()
