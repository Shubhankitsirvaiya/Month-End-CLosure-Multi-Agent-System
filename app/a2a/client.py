"""A2A JSON-RPC client for the separately hosted valuation specialist.

The orchestrator should not know how HTTP, JSON-RPC, Agent Cards, or artifacts
work. This adapter hides transport details and returns the same ``AgentOutput``
shape produced by local specialists.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx

from app.core.config import Settings, get_settings


class A2AClientError(RuntimeError):
    """Raised when the remote A2A contract cannot complete a task."""


class ValuationA2AClient:
    """Discover and call the valuation agent through its public A2A contract."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Create a client.

        Args:
            settings: Optional dependency-injected settings. Passing it makes
                tests reproducible; omitting it uses the cached application
                configuration.
        """
        self.settings = settings or get_settings()
        # Removing a trailing slash prevents accidental ``//`` in endpoint URLs.
        self.base_url = self.settings.valuation_agent_url.rstrip("/")

    async def get_agent_card(self) -> dict[str, Any]:
        """Fetch the Agent Card that advertises identity, skills, and version.

        ``raise_for_status`` converts non-2xx responses into an exception so the
        orchestrator can use its explicit local fallback policy.
        """
        # Agent Card discovery should be fast. A one-second connection budget and
        # four-second total budget trigger continuity fallback promptly when the
        # remote service is unavailable.
        discovery_timeout = httpx.Timeout(4.0, connect=1.0)
        async with httpx.AsyncClient(timeout=discovery_timeout) as client:
            response = await client.get(f"{self.base_url}/.well-known/agent-card.json")
            response.raise_for_status()
            return response.json()

    async def run_valuation_task(
        self,
        *,
        run_id: str,
        portfolio_id: str | None,
        close_period: str,
        task: str,
        existing_exceptions: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Submit one valuation task and return its artifact plus trace metadata.

        Args:
            run_id: Correlation ID of the parent close review.
            portfolio_id: Optional portfolio scope; ``None`` means all portfolios.
            close_period: Accounting date being reviewed.
            task: Natural-language objective assigned by the supervisor.
            existing_exceptions: Prior findings supplied as investigation context.

        Returns:
            ``(output, trace)`` where output is the normalized agent artifact and
            trace records the remote task/context IDs needed for auditability.
        """

        card = await self.get_agent_card()
        # JSON-RPC request IDs correlate transport responses; message IDs identify
        # the semantic A2A message. They intentionally serve different purposes.
        request_id = f"req-{uuid.uuid4().hex}"
        message_id = f"msg-{uuid.uuid4().hex}"
        payload_data = {
            "run_id": run_id,
            "portfolio_id": portfolio_id,
            "close_period": close_period,
            "task": task,
            "existing_exceptions": existing_exceptions,
        }
        # A2A uses JSON-RPC 2.0. ``message/send`` requests that the remote agent
        # accept and execute a task represented as message parts.
        rpc_payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": message_id,
                    "parts": [{"kind": "text", "text": json.dumps(payload_data)}],
                }
            },
        }
        headers = {"A2A-Version": self.settings.a2a_version, "Content-Type": "application/json"}
        # 120 seconds is longer than discovery because task execution may include
        # retrieval, tool calls, and model generation.
        # Execution may legitimately take longer, but connection establishment is
        # still bounded so an offline endpoint does not stall the entire review.
        task_timeout = httpx.Timeout(120.0, connect=2.0)
        async with httpx.AsyncClient(timeout=task_timeout) as client:
            response = await client.post(self.base_url + "/", json=rpc_payload, headers=headers)
        if response.status_code >= 400:
            raise A2AClientError(f"A2A HTTP {response.status_code}: {response.text[:500]}")
        body = response.json()
        if "error" in body:
            raise A2AClientError(f"A2A JSON-RPC error: {body['error']}")
        task_object = body.get("result", {})
        # Agent output is delivered as an artifact rather than inferred from a
        # status message. Missing artifacts are therefore a contract failure.
        artifacts = task_object.get("artifacts", [])
        if not artifacts:
            raise A2AClientError("Valuation agent completed without an artifact.")
        artifact_text = artifacts[0]["parts"][0]["text"]
        output = json.loads(artifact_text)
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

