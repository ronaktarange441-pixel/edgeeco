"""
EcoEdge Autonomous Agent - Hackathon Prototype (Serverless Edition)
====================================================================
SERVERLESS-SAFE REWRITE - no background threads, no while-loops, no
long-lived in-memory state. (Vercel serverless functions freeze between
invocations, so a background simulator thread would never run there.)

Every call to GET /api/state:
  1. Generates a fresh mock sensor reading on the fly with Python `random`:
       - Temperature : uniform 70.0 - 110.0 degC
       - Vibration   : uniform 2.0 - 18.0 mm/s
  2. Sends that reading to a Google Gemini agent for a JSON decision.
  3. Falls back to a deterministic rule engine when the Gemini key is
     missing or the API call fails - so the endpoint ALWAYS returns a
     valid response (no more "ERR" on the dashboard).

Security: GEMINI_API_KEY is read server-side from Vercel env vars and is
never exposed in any API response or static asset.
"""

import json
import logging
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("ecoedge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --------------------------------------------------------------------------
# Config & secrets
# --------------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"
MACHINE = "PNEUMATIC_PRESS_03"
START_TIME = time.time()

SYSTEM_PROMPT = (
    "You are an Industrial Safety Agent for a smart factory edge controller.\n"
    "You receive live IoT sensor telemetry from one machine. Analyse the reading and decide\n"
    "the machine's safety state and the action to take.\n\n"
    "Safety rules (they are YOUR knowledge of machine-health limits):\n"
    "- Temperature below 85 degC is NORMAL.\n"
    "- Temperature 85-94 degC is HIGH (warn / throttle the machine).\n"
    "- Temperature 95+ degC is CRITICAL (emergency shutdown).\n"
    "- Vibration below 6.0 mm/s is NORMAL.\n"
    "- Vibration 6.0-9.9 mm/s is HIGH (warn / throttle the machine).\n"
    "- Vibration 10.0+ mm/s is CRITICAL (emergency shutdown).\n\n"
    'Respond with ONLY a single valid JSON object, no markdown fences, no commentary:\n'
    '{"status": "SAFE"|"CRITICAL", "action": "NONE"|"THROTTLE"|"EMERGENCY_SHUTDOWN",'
    ' "reason": "short human-readable reason or NONE"}\n'
    "- SAFE + NONE when every reading is normal.\n"
    "- CRITICAL + THROTTLE when a HIGH threshold is exceeded.\n"
    "- CRITICAL + EMERGENCY_SHUTDOWN when a CRITICAL threshold is exceeded."
)
# --------------------------------------------------------------------------
# Gemini client - built ONLY when the key exists. A missing key must never
# crash the app; the rule engine below keeps the dashboard alive.
# --------------------------------------------------------------------------
CLIENT = None
GEMINI_MODEL = "gemini-3.8-flash"

if GEMINI_API_KEY:
    try:
        from google import genai
        from google.genai import types

        CLIENT = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=15_000),
        )
    except Exception as exc:  # pragma: no cover - SDK problems degrade to rules
        log.warning("Gemini SDK init failed (%s) - rule fallback only", repr(exc))
else:
    log.warning("GEMINI_API_KEY not set - serving with rule-engine fallback only.")


# --------------------------------------------------------------------------
# Deterministic rule engine: same thresholds as the Gemini system prompt.
# Guarantees a decision on every request, no network required.
# --------------------------------------------------------------------------
def rule_fallback(temp: float, vib: float):
    if temp >= 95.0 or vib >= 10.0:
        return "CRITICAL", "EMERGENCY_SHUTDOWN"
    if temp >= 85.0 or vib >= 6.0:
        return "CRITICAL", "THROTTLE"
    return "SAFE", "NONE"


def ask_gemini(payload: dict) -> dict:
    """Ask Gemini for a decision. On ANY failure (no key, timeout, network,
    malformed reply) fall back to the rule engine so every request succeeds."""
    prompt = (
        f"Machine: {payload['machine']}\n"
        f"Temperature: {payload['temperature']} degC\n"
        f"Vibration: {payload['vibration']} mm/s\n"
        "Respond with JSON only."
    )
    started = time.time()
    source = "gemini"
    try:
        if CLIENT is None:
            raise RuntimeError("GEMINI_API_KEY missing")
        response = CLIENT.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        raw = (response.text or "").strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        decision = {
            "status": str(parsed.get("status", "")).upper(),
            "action": str(parsed.get("action", "")).upper(),
            "reason": str(parsed.get("reason", "") or "NONE"),
        }
        if decision["action"] not in {"NONE", "THROTTLE", "EMERGENCY_SHUTDOWN"}:
            raise ValueError("unknown action")
    except Exception as exc:  # noqa: BLE001 - any failure degrades to rules
        log.warning("Gemini evaluation failed (%s) - using rule engine", repr(exc))
        status, action = rule_fallback(payload["temperature"], payload["vibration"])
        decision = {
            "status": status,
            "action": action,
            "reason": "AI unavailable (%s) - rule engine" % type(exc).__name__,
        }
        source = "rules"
    decision["latency_ms"] = int((time.time() - started) * 1000)
    decision["source"] = source
    decision["model"] = GEMINI_MODEL
    return decision
# --------------------------------------------------------------------------
# FastAPI app - no background loops, no startup thread, fully serverless
# --------------------------------------------------------------------------
app = FastAPI(title="EcoEdge Autonomous Agent (Serverless)")


@app.get("/api/health")
def api_health() -> dict:
    """Quick liveness probe (handy for debugging the Vercel deployment)."""
    return {
        "ok": True,
        "gemini_key_set": bool(GEMINI_API_KEY),
        "machine": MACHINE,
        "uptime_s": round(time.time() - START_TIME, 1),
    }


@app.get("/api/state")
def api_state() -> dict:
    """One request = one fresh mock reading + one AI decision.
    Fully regenerated per call - no persistent state required."""
    temperature = round(random.uniform(70.0, 110.0), 1)
    vibration = round(random.uniform(2.0, 18.0), 2)
    now = time.strftime("%H:%M:%S")

    payload = {
        "machine": MACHINE,
        "temperature": temperature,
        "vibration": vibration,
    }
    decision = ask_gemini(payload)

    return {
        "sensor": {
            "machine": MACHINE,
            "temperature": temperature,
            "vibration": vibration,
            "timestamp": now,
        },
        "agent": {
            "status": decision.get("status", "SAFE"),
            "action": decision.get("action", "NONE"),
            "reason": decision.get("reason") or "NONE",
            "source": decision.get("source", "rules"),
            "latency_ms": decision.get("latency_ms", 0),
            "last_eval": now,
            "model": decision.get("model", GEMINI_MODEL),
        },
        "log": [
            {
                "t": now,
                "temp": temperature,
                "vib": vibration,
                "action": decision.get("action", "NONE"),
            }
        ],
        "uptime_s": round(time.time() - START_TIME, 1),
    }


# Serve the dashboard for local runs (Vercel serves /public via rewrites).
app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
