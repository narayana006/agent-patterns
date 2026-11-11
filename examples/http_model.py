"""Plug a real model into any pattern: a Model is just ``prompt -> text``.

This adapter talks to any OpenAI-compatible chat completions endpoint (a
hosted API or a model server running on your own machine) using only the
standard library. Nothing in ``patterns/`` imports it.

    export MODEL_BASE_URL=http://localhost:8080/v1
    export MODEL_NAME=my-model
    export MODEL_API_KEY=...        # optional, sent as a bearer token
    python examples/http_model.py "The app will not sync my notes."

The command above sends the message through the router pattern.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path


def http_model(base_url: str, model_name: str, api_key: str | None = None, timeout: float = 60.0):
    """Return a ``prompt -> text`` function backed by ``POST {base_url}/chat/completions``."""

    def call(prompt: str) -> str:
        body = json.dumps({"model": model_name, "messages": [{"role": "user", "content": prompt}]}).encode()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        return payload["choices"][0]["message"]["content"]

    return call


def main(argv: list[str]) -> int:
    base_url, model_name = os.environ.get("MODEL_BASE_URL"), os.environ.get("MODEL_NAME")
    if not base_url or not model_name or len(argv) != 2:
        print("usage: MODEL_BASE_URL=... MODEL_NAME=... python examples/http_model.py \"customer message\"", file=sys.stderr)
        return 2
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "patterns"))
    import router

    router.route(argv[1], http_model(base_url, model_name, os.environ.get("MODEL_API_KEY")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
