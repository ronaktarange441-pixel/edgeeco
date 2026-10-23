"""
EcoEdge API - unit tests.

Run with either:
    pytest test_app.py
or (pytest is optional; the tests run serially):
    python test_app.py

The tests exercise the FastAPI routes through Starlette's TestClient and
the rule-based fallback directly. They require no network access and no
GEMINI_API_KEY (the Gemini client is mocked / bypassed).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when running from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import api.index as api  # noqa: E402  (import order enforced by the sys.path fix)
from fastapi.testclient import TestClient  # noqa: E402


def _fixed_decision(*args: object, **kwargs: object) -> dict[str, object]:
    """Deterministic stand-in for ``ask_gemini`` used to keep tests hermetic.

    Returns a SAFE / NONE decision labelled as coming from the rule engine,
    shaped exactly like a real Gemini decision dictionary.
    """
    return {
        "status": "SAFE",
        "action": "NONE",
        "reason": "mocked decision",
        "source": "rules",
        "latency_ms": 1,
        "model": api.GEMINI_MODEL,
    }


def test_api_health_returns_200() -> None:
    """GET /api/health reports ok=True, the machine name and uptime."""
    with TestClient(api.app) as client:
        response = client.get("/api/health")

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["machine"] == api.MACHINE
        assert body["uptime_s"] >= 0
        assert isinstance(body["gemini_key_set"], bool)


def test_api_state_returns_valid_schema() -> None:
    """GET /api/state returns the full telemetry + decision payload."""
    original = api.ask_gemini
    api.ask_gemini = _fixed_decision
    try:
        with TestClient(api.app) as client:
            response = client.get("/api/state")

            assert response.status_code == 200
            body = response.json()

            sensor = body["sensor"]
            assert sensor["machine"] == api.MACHINE
            assert 70.0 <= sensor["temperature"] <= 110.0
            assert 2.0 <= sensor["vibration"] <= 18.0
            assert sensor["timestamp"]

            agent = body["agent"]
            assert agent["status"] == "SAFE"
            assert agent["action"] == "NONE"
            assert agent["source"] == "rules"
            assert isinstance(agent["latency_ms"], int)
            assert agent["model"] == api.GEMINI_MODEL

            assert isinstance(body["uptime_s"], (int, float))
            assert body["uptime_s"] >= 0

            assert len(body["log"]) == 1
            entry = body["log"][0]
            assert {"t", "temp", "vib", "action"} <= set(entry)
    finally:
        api.ask_gemini = original


def test_rule_fallback_thresholds() -> None:
    """The deterministic rule engine matches the documented thresholds."""
    assert api.rule_fallback(96.0, 3.0) == ("CRITICAL", "EMERGENCY_SHUTDOWN")
    assert api.rule_fallback(80.0, 11.0) == ("CRITICAL", "EMERGENCY_SHUTDOWN")
    assert api.rule_fallback(88.0, 3.0) == ("CRITICAL", "THROTTLE")
    assert api.rule_fallback(80.0, 7.0) == ("CRITICAL", "THROTTLE")
    assert api.rule_fallback(80.0, 3.0) == ("SAFE", "NONE")


def test_ask_gemini_falls_back_to_rules_without_client() -> None:
    """Without a Gemini client, ask_gemini degrades to the rule engine."""
    original = api.CLIENT
    api.CLIENT = None
    try:
        decision = api.ask_gemini(
            {"machine": api.MACHINE, "temperature": 99.0, "vibration": 12.0}
        )
        assert decision["status"] == "CRITICAL"
        assert decision["action"] == "EMERGENCY_SHUTDOWN"
        assert decision["source"] == "rules"
        assert "rule engine" in decision["reason"]
    finally:
        api.CLIENT = original


if __name__ == "__main__":
    # Allow the tests to run even when pytest is not installed.
    test_api_health_returns_200()
    test_api_state_returns_valid_schema()
    test_rule_fallback_thresholds()
    test_ask_gemini_falls_back_to_rules_without_client()
    print("All 4 tests passed.")