"""Modular A2A client used by the backend and notebook."""
from __future__ import annotations

import json
import uuid
from typing import Any

import httpx

from app.core.config import Settings, get_settings


class A2AClientError(RuntimeError):
    """Raised when the remote A2A contract cannot complete a task."""


class AgentCardDiscoveryMixin:
    """Fast capability discovery behavior."""

    async def get_agent_card(self) -> dict[str, Any]:
        """Fetch the identity, version, endpoint, and advertised skills."""
        # Discovery has a smaller timeout than execution so offline fallback starts
        # quickly instead of consuming the month-end processing window.
        timeout = httpx.Timeout(4.0, connect=1.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{self.base_url}/.well-known/agent-card.json")
            response.raise_for_status()
            return response.json()


class ValuationTaskMixin:
    """JSON-RPC message/send task behavior."""

    async def run_valuation_task(
        self,
        *,
        run_id: str,
        portfolio_id: str | None,
        close_period: str,
        task: str,
        existing_exceptions: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Send one task and return normalized output plus remote trace IDs."""
        card = await self.get_agent_card()
        request_id = f"req-{uuid.uuid4().hex}"
        message_id = f"msg-{uuid.uuid4().hex}"
        request_data = {
            "run_id": run_id, "portfolio_id": portfolio_id,
            "close_period": close_period, "task": task,
            "existing_exceptions": existing_exceptions,
        }
        # JSON-RPC ID correlates transport; messageId identifies semantic message.
        rpc_payload = {
            "jsonrpc": "2.0", "id": request_id, "method": "message/send",
            "params": {"message": {
                "role": "user", "messageId": message_id,
                "parts": [{"kind": "text", "text": json.dumps(request_data)}],
            }},
        }
        headers = {"A2A-Version": self.settings.a2a_version, "Content-Type": "application/json"}
        timeout = httpx.Timeout(120.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self.base_url + "/", json=rpc_payload, headers=headers)
        if response.status_code >= 400:
            raise A2AClientError(f"A2A HTTP {response.status_code}: {response.text[:500]}")
        body = response.json()
        if "error" in body:
            raise A2AClientError(f"A2A JSON-RPC error: {body['error']}")
        task_object = body.get("result", {})
        artifacts = task_object.get("artifacts", [])
        if not artifacts:
            raise A2AClientError("Valuation agent completed without an artifact.")
        output = json.loads(artifacts[0]["parts"][0]["text"])
        trace = {
            "agent_name": card.get("name", "Remote Valuation Agent"),
            "agent_version": card.get("version"),
            "task_id": task_object.get("id"),
            "context_id": task_object.get("contextId"),
            "state": task_object.get("status", {}).get("state"),
            "request_id": request_id,
            "endpoint": self.base_url,
        }
        return output, trace


class ValuationA2AClient(AgentCardDiscoveryMixin, ValuationTaskMixin):
    """Configured client combining discovery and task behavior."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Store configuration and normalize the remote service base URL."""
        # Explicit settings aid tests; rstrip prevents accidental double slashes.
        self.settings = settings or get_settings()
        self.base_url = self.settings.valuation_agent_url.rstrip("/")
