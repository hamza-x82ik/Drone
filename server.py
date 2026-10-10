"""Tiny standard-library web server for Drone Commander.

    python server.py            # http://127.0.0.1:8000

Endpoints
    GET  /api/world    map, rules and the active model name
    GET  /api/health   is Ollama reachable and is the model installed
    POST /api/plan     {"command": str, "state": {...}} -> plan + safety verdict
    GET  /api/audit    the last 50 audit-log entries
"""
from __future__ import annotations

import json
import mimetypes
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import planner
import validator

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
AUDIT_FILE = Path(os.environ.get("AUDIT_FILE", ROOT / "logs" / "audit.jsonl"))
MAX_BODY = 4096
MAX_COMMAND = 300

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
}


def load_dotenv(path=ROOT / ".env"):
    """Load KEY=VALUE lines from .env. Real environment variables win over the file;
    if a key appears twice in the file, the later line wins."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    values = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    for key, value in values.items():
        os.environ.setdefault(key, value)


def clean_state(raw):
    """Accept only a well-formed drone state from the browser."""
    try:
        x, y, battery = float(raw["x"]), float(raw["y"]), float(raw["battery"])
        flying = raw["flying"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("state needs numeric x, y, battery and boolean flying") from exc
    size = validator.WORLD["size"]
    if not isinstance(flying, bool) or not (0 <= x <= size and 0 <= y <= size and 0 <= battery <= 100):
        raise ValueError("state values are out of range")
    return {"x": x, "y": y, "flying": flying, "battery": battery}


def audit(entry):
    try:
        AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # never let logging break a request


def handle_plan(body):
    command = body.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    command = command.strip()
    if len(command) > MAX_COMMAND:
        raise ValueError(f"command is longer than {MAX_COMMAND} characters")
    state = clean_state(body.get("state"))
    backend = body.get("backend")
    if backend is not None and backend not in planner.BACKENDS:
        raise ValueError("backend must be one of: " + ", ".join(planner.BACKENDS))

    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "command": command, "state": state, "backend": backend or "default"}
    try:
        proposal = planner.plan(command, state, backend=backend)
    except planner.PlannerError as exc:
        entry.update(outcome="error", error=str(exc))
        audit(entry)
        return 502, {"outcome": "error", "error": str(exc)}

    result = {
        "model": proposal["model"], "attempts": proposal["attempts"],
        "latency_ms": proposal["latency_ms"], "raw": proposal["raw"],
        "steps": proposal["steps"], "clarification": proposal["clarification"],
        "note": proposal.get("note"), "validation": None,
    }
    if not proposal["understood"] or not proposal["steps"]:
        result["outcome"] = "clarify"
        if not result["clarification"]:
            result["clarification"] = "I could not turn that into drone steps. Try naming a place or a direction and distance."
    else:
        verdict = validator.validate(proposal["steps"], state)
        result["validation"] = verdict
        result["outcome"] = "accepted" if verdict["ok"] else "rejected"

    entry.update(outcome=result["outcome"], model=result["model"], note=result["note"], steps=proposal["steps"])
    if result["validation"] and not result["validation"]["ok"]:
        bad = next((s for s in result["validation"]["steps"] if s["status"] == "rejected"), None)
        entry["rejection"] = bad["reason"] if bad else result["validation"]["plan_error"]
    audit(entry)
    return 200, result


class Handler(BaseHTTPRequestHandler):
    server_version = "DroneCommander/1.0"

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, code, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for key, value in SECURITY_HEADERS.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/world":
            return self._send(200, {"world": validator.WORLD, "rules": validator.RULES, "model": planner.config()["label"]})
        if path == "/api/health":
            return self._send(200, planner.health())
        if path == "/api/audit":
            try:
                lines = AUDIT_FILE.read_text(encoding="utf-8").splitlines()[-50:]
                return self._send(200, {"entries": [json.loads(line) for line in lines]})
            except (OSError, ValueError):
                return self._send(200, {"entries": []})
        return self._static(path)

    def _static(self, path):
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (STATIC / rel).resolve()
        if STATIC not in target.parents or not target.is_file():
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/plan":
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if not 0 < length <= MAX_BODY:
            return self._send(413, {"outcome": "error", "error": "request body missing or too large"})
        try:
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            code, payload = handle_plan(body)
        except (ValueError, json.JSONDecodeError) as exc:
            return self._send(400, {"outcome": "error", "error": str(exc)})
        self._send(code, payload)


def make_server(host="127.0.0.1", port=8000):
    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    load_dotenv()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    cfg = planner.config()
    print(f"Drone Commander on http://{host}:{port}")
    print(f"Default mode: {cfg['default_mode']}  (switch live in the browser)")
    print(f"  API   : {cfg['api_label']}, key {'set' if cfg['key'] else 'MISSING'}")
    if cfg["key"] and not cfg["api_model"]:
        print("          No API model set. Run: python tools/list_models.py")
    print(f"  Local : {cfg['local_model']} via Ollama, fallback {'on' if cfg['fallback'] else 'off'}")
    try:
        make_server(host, port).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
