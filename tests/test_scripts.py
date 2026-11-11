"""Every pattern must run as a script, offline, and stay small."""

import ast
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = sorted((ROOT / "patterns").glob("*.py"))
STDLIB = set(sys.stdlib_module_names)


def test_all_six_patterns_exist():
    assert [p.stem for p in PATTERNS] == [
        "evaluator_optimizer", "human_in_the_loop", "memory", "planner_executor", "router", "tool_retry",
    ]


@pytest.mark.parametrize("path", PATTERNS, ids=lambda p: p.stem)
def test_runs_as_a_script_and_prints_a_trace(path):
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=30, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert sum(line.startswith("[") for line in result.stdout.splitlines()) >= 6


@pytest.mark.parametrize("path", PATTERNS, ids=lambda p: p.stem)
def test_under_200_lines_and_standard_library_only(path):
    source = path.read_text()
    assert len(source.splitlines()) < 200
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= STDLIB, f"non-stdlib imports: {imported - STDLIB}"
    assert not imported & {"socket", "urllib", "http", "requests"}, "patterns must not touch the network"


def test_interactive_approval_reads_from_stdin():
    script = ROOT / "patterns" / "human_in_the_loop.py"
    result = subprocess.run([sys.executable, str(script), "--interactive"], input="n\ntoo much\ny\n",
                            capture_output=True, text=True, timeout=30, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "denied: too much" in result.stdout
    assert "side effects: ['refund 39.00 on A-1042']" in result.stdout


def test_http_model_adapter_against_a_local_fake_server():
    import http_model

    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": "technical"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = http_model.http_model(f"http://127.0.0.1:{server.server_port}/v1/", "test-model", api_key="test-key")
        assert model("classify this") == "technical"
    finally:
        server.shutdown()
        server.server_close()
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"] == {"model": "test-model", "messages": [{"role": "user", "content": "classify this"}]}
