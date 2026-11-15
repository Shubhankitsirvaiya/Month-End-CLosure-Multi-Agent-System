"""A2A server for the independently hosted LangGraph valuation agent.

The service intentionally exposes only the agent's public skills and task
artifacts. Its internal graph, prompts, tools, and memory remain private.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException

from app.agents.specialists_modular import build_specialist_agent
from app.core.config import get_settings


# Build the graph once at service startup. Rebuilding it per request would waste
# initialization time and repeatedly recreate retrieval resources.
settings = get_settings()
valuation_graph = build_specialist_agent("valuation", settings)
# In-memory storage is sufficient for the classroom deployment. A production
# service would replace this dictionary with durable, TTL-managed task storage.
task_store: dict[str, dict[str, Any]] = {}

app = FastAPI(
    title="Meridian Ridge Valuation A2A Agent",
    description="Independent LangGraph valuation specialist for month-end pricing exceptions.",
    version="1.0.0",
)


@app.get("/.well-known/agent-card.json")
async def agent_card() -> dict[str, Any]:
    """Publish the discoverable A2A capability contract.

    The card plays a role similar to a tool description, but for an entire
    agent: identity, skills, endpoint, transport, protocol version, and modes.
    """

    return {
        "name": "Meridian Ridge Valuation Agent",
        "description": "Investigates stale prices, source conflicts, fair-value evidence, and NAV impact.",
        "version": "1.0.0",
        "protocolVersion": settings.a2a_version,
        "preferredTransport": "JSONRPC",
        "url": settings.valuation_agent_url.rstrip("/") + "/",
        "supportedInterfaces": [{
            "url": settings.valuation_agent_url.rstrip("/") + "/",
            "protocolBinding": "JSONRPC",
            "protocolVersion": settings.a2a_version,
        }],
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text", "application/json"],
        "defaultOutputModes": ["text", "application/json"],
        "skills": [
            {
                "id": "investigate_valuation_exception",
                "name": "Investigate valuation exception",
                "description": "Compares approved sources, detects stale prices, applies policy, and calculates NAV impact.",
                "tags": ["valuation", "pricing", "stale-price", "NAV", "month-end"],
                "examples": ["Investigate stale price for BD009 in MR-CREDIT-HY at 2026-10-31"],
            }
        ],
    }


@app.get("/ping")
async def ping() -> dict[str, str]:
    """Return a cheap liveness response for Docker and operator checks."""
    return {"status": "Healthy"}


def _rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    """Create a JSON-RPC 2.0 error while preserving request correlation.

    Args:
        request_id: Original caller-supplied JSON-RPC ID.
        code: Standard or application-specific numeric error code.
        message: Safe diagnostic text; internal stack traces are not exposed.
    """
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


@app.post("/")
async def a2a_rpc(payload: dict[str, Any], a2a_version: str | None = Header(default=None)) -> dict[str, Any]:
    """Handle supported A2A JSON-RPC methods.

    Args:
        payload: Parsed JSON-RPC request body. ``message/send`` starts work and
            ``tasks/get`` retrieves a previously stored task.
        a2a_version: Optional ``A2A-Version`` HTTP header injected by FastAPI.

    Returns:
        A JSON-RPC result or error object. Protocol errors remain HTTP 200 because
        JSON-RPC communicates method-level failure in the response body.
    """
    request_id = payload.get("id")
    method = payload.get("method")
    if a2a_version and not a2a_version.startswith(("1.0", "0.3")):
        return _rpc_error(request_id, -32060, f"Unsupported A2A version: {a2a_version}")

    # Accept both current and legacy method spellings to make the demo tolerant
    # of clients using adjacent A2A specification versions.
    if method in {"tasks/get", "GetTask"}:
        task_id = payload.get("params", {}).get("id")
        if task_id not in task_store:
            return _rpc_error(request_id, -32001, "Task not found")
        return {"jsonrpc": "2.0", "id": request_id, "result": task_store[task_id]}

    if method not in {"message/send", "SendMessage"}:
        return _rpc_error(request_id, -32601, f"Unsupported method: {method}")

    # A2A message parts may support several media types. This project explicitly
    # accepts one JSON-encoded text part to keep validation and teaching clear.
    try:
        message = payload["params"]["message"]
        text_part = next(part["text"] for part in message["parts"] if part.get("kind") == "text")
        request_data = json.loads(text_part)
    except (KeyError, StopIteration, json.JSONDecodeError) as exc:
        return _rpc_error(request_id, -32602, f"Invalid valuation task message: {exc}")

    # task_id identifies remote execution; context_id links it to the parent
    # month-end run across service boundaries.
    task_id = f"a2a-val-{uuid.uuid4().hex[:12]}"
    context_id = request_data.get("run_id") or f"ctx-{uuid.uuid4().hex[:12]}"
    task_store[task_id] = {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": "TASK_STATE_WORKING"},
        "artifacts": [],
    }
    try:
        # ``ainvoke`` does not block the event loop while the agent performs
        # asynchronous model or retrieval calls.
        result = await valuation_graph.ainvoke({
            "run_id": request_data.get("run_id"),
            "portfolio_id": request_data.get("portfolio_id"),
            "close_period": request_data.get("close_period", "2026-10-31"),
            "task": request_data.get("task", "Investigate all valuation exceptions."),
            "existing_exceptions": request_data.get("existing_exceptions", []),
        })
        output = result["output"]
        task_object = {
            "id": task_id,
            "contextId": context_id,
            "status": {"state": "TASK_STATE_COMPLETED"},
            "artifacts": [{
                "artifactId": f"artifact-{uuid.uuid4().hex[:12]}",
                "name": "valuation_review",
                "description": "Structured valuation-agent findings and policy evidence.",
                "parts": [{"kind": "text", "text": json.dumps(output)}],
            }],
            "metadata": {"agent": "valuation", "run_id": request_data.get("run_id")},
        }
        task_store[task_id] = task_object
        return {"jsonrpc": "2.0", "id": request_id, "result": task_object}
    except Exception as exc:  # The RPC boundary must translate internal exceptions.
        task_store[task_id]["status"] = {"state": "TASK_STATE_FAILED", "message": {"role": "agent", "parts": [{"kind": "text", "text": str(exc)}]}}
        return _rpc_error(request_id, -32603, f"Valuation agent failed: {exc}")


@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict[str, Any]:
    """Convenience REST endpoint for humans; A2A clients use ``tasks/get``."""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_store[task_id]

