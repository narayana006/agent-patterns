"""Memory: a short-term window for the conversation, a long-term store for facts.

The idea
    Short-term memory is the recent conversation. It is kept to a fixed
    number of turns; older turns are folded into a running summary so the
    prompt stops growing. Long-term memory is a small store of facts that
    outlives the conversation (here a JSON file). Before each reply the facts
    relevant to the new message are looked up and added to the prompt, and
    the model can save a new fact by ending its reply with "REMEMBER: ...".

Run it:  python patterns/memory.py
"""

import json
import tempfile
from pathlib import Path
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


# --- long-term memory: facts on disk, recalled by word overlap ------------------

class FactStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.facts: list[str] = json.loads(self.path.read_text()) if self.path.exists() else []

    def save(self, fact: str) -> None:
        if fact not in self.facts:
            self.facts.append(fact)
            self.path.write_text(json.dumps(self.facts, indent=2))

    def recall(self, query: str, limit: int = 2) -> list[str]:
        """Facts sharing the most words with the query. Swap in embeddings here."""
        words = {w.strip(".,?!'").lower() for w in query.split()}
        words = {w for w in words if len(w) > 3}
        scored = [(len(words & {w.strip(".,").lower() for w in fact.split()}), fact) for fact in self.facts]
        return [fact for score, fact in sorted(scored, key=lambda pair: -pair[0]) if score > 0][:limit]


# --- short-term memory: last few turns verbatim, the rest summarised ------------

class Conversation:
    def __init__(self, model: Model, store: FactStore, window: int = 2):
        self.model, self.store, self.window = model, store, window
        self.turns: list[tuple[str, str]] = []  # (user, assistant)
        self.summary = ""

    def _compact(self) -> None:
        """Fold turns that fell out of the window into the running summary."""
        while len(self.turns) > self.window:
            user, assistant = self.turns.pop(0)
            self.summary = self.model(
                f"Summary so far: {self.summary or '(nothing yet)'}\n"
                f"Add this exchange in one sentence.\nUser: {user}\nAssistant: {assistant}"
            ).strip()
            trace("summary", self.summary)

    def build_prompt(self, message: str) -> str:
        facts = self.store.recall(message)
        for fact in facts:
            trace("recall", fact)
        parts = []
        if facts:
            parts.append("Known facts about the user:\n" + "\n".join(f"- {fact}" for fact in facts))
        if self.summary:
            parts.append(f"Earlier in this conversation: {self.summary}")
        parts += [f"User: {user}\nAssistant: {assistant}" for user, assistant in self.turns]
        parts.append(f"User: {message}\nAssistant:")
        return "\n\n".join(parts)

    def send(self, message: str) -> str:
        trace("user", message)
        reply = self.model(self.build_prompt(message)).strip()
        if "REMEMBER:" in reply:
            reply, fact = (part.strip() for part in reply.split("REMEMBER:", 1))
            self.store.save(fact)
            trace("remember", fact)
        trace("reply", reply)
        self.turns.append((message, reply))
        self._compact()
        return reply


def demo(path: Path | None = None) -> list[str]:
    path = path or Path(tempfile.mkdtemp()) / "facts.json"
    print("--- session 1 ---")
    model = ScriptedModel([
        "Noted, I will keep suggestions vegetarian. REMEMBER: The user is vegetarian.",
        "Welcome! The Bolhao market is the place to start. REMEMBER: The user lives in Porto.",
        "Try a chickpea and spinach stew; it takes about 25 minutes.",
        "The user is vegetarian and asked for that to be remembered.",  # summary after turn 3
        "Any stall in the Bolhao market sells dried chickpeas by weight.",
        "The user is vegetarian, has moved to Porto and was pointed to the Bolhao market.",  # after turn 4
    ])
    chat = Conversation(model, FactStore(path), window=2)
    chat.send("I'm vegetarian, please remember that.")
    chat.send("I just moved to Porto, any tips for food markets?")
    chat.send("What vegetarian dinner can I cook tonight in under 30 minutes?")
    chat.send("Where in Porto can I buy chickpeas?")
    print(f"\nturns kept verbatim: {len(chat.turns)} of 4; the other 2 live on in the summary")

    print("\n--- session 2 (new conversation, same fact store) ---")
    model = ScriptedModel(["You are in Porto and vegetarian, so try the vegetable cataplana at a tasca near the river."])
    chat = Conversation(model, FactStore(path), window=2)
    reply = chat.send("Suggest somewhere for a vegetarian dinner in Porto.")
    print("\nprompt sent to the model in session 2:")
    print("\n".join(f"    {line}" for line in model.prompts[0].splitlines()))
    return [reply, *FactStore(path).facts]


if __name__ == "__main__":
    demo()
