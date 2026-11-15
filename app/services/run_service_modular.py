"""Modular operations run service used by FastAPI and notebook."""
from __future__ import annotations

import asyncio
import traceback
import uuid
from typing import Any

from langgraph.types import Command

from app.agents.orchestrator_modular import build_orchestrator_graph
from app.core.config import Settings, get_settings
from app.core.models import ApprovalDecisionRequest, RunCreateRequest, utc_now_iso
from app.evaluation.evaluator import evaluate_run
from app.storage.repository import RunRepository


def jsonable(value: Any) -> Any:
    """Recursively convert Pydantic/tuple values into JSON-safe structures."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


class RunExecutionMixin:
    """Create, invoke, interrupt-project, and resume behavior."""

    def create(self, request: RunCreateRequest) -> dict[str, Any]:
        """Persist a queued run and schedule graph work on the current event loop."""
        run_id = f"close-{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        record = {
            "run_id": run_id, "scenario_id": request.scenario_id,
            "portfolio_id": request.portfolio_id, "close_period": request.close_period,
            "user_goal": request.user_goal, "status": "queued",
            "created_at": now, "updated_at": now, "state": {},
            "evaluation": None, "error": None,
        }
        self.repository.save(record)
        # create_task lets the API return HTTP 202 while agents continue.
        self._tasks[run_id] = asyncio.create_task(self._run_initial(record))
        return record

    async def _run_initial(self, record: dict[str, Any]) -> None:
        """Translate persisted request fields into initial shared graph state."""
        initial_state = {
            "run_id": record["run_id"], "scenario_id": record["scenario_id"],
            "portfolio_id": record["portfolio_id"], "close_period": record["close_period"],
            "user_goal": record["user_goal"],
        }
        await self._invoke(record["run_id"], initial_state)

    async def _invoke(self, run_id: str, graph_input: Any) -> None:
        """Invoke/resume the graph and persist its operator-facing projection."""
        try:
            self.repository.update(run_id, status="running", updated_at=utc_now_iso())
            config = {"configurable": {"thread_id": run_id}}
            result = jsonable(await self.graph.ainvoke(graph_input, config=config))
            interrupts = result.pop("__interrupt__", []) if isinstance(result, dict) else []
            if interrupts:
                first = interrupts[0]
                result["approval_request"] = jsonable(getattr(first, "value", first))
                result["approval_required"] = True
                result["status"] = "approval_pending"
            self.repository.update(
                run_id, status=result.get("status", "completed"),
                state=result, updated_at=utc_now_iso(),
            )
        except Exception as exc:
            # Background failures must be persisted because no request is awaiting them.
            self.repository.update(
                run_id, status="failed", updated_at=utc_now_iso(),
                error={"type": type(exc).__name__, "message": str(exc),
                       "traceback": traceback.format_exc(limit=8)},
            )

    def resume(self, run_id: str, decision: ApprovalDecisionRequest) -> dict[str, Any]:
        """Resume only a run that is currently waiting for approval."""
        record = self.require(run_id)
        if record["status"] != "approval_pending":
            raise ValueError("This run is not waiting for approval")
        payload = decision.model_dump()
        payload["decided_at"] = utc_now_iso()
        self._tasks[run_id] = asyncio.create_task(self._invoke(run_id, Command(resume=payload)))
        return self.repository.update(run_id, status="resuming", updated_at=utc_now_iso())


class RunQueryMixin:
    """Read, list, evaluate, and event-projection behavior."""

    def require(self, run_id: str) -> dict[str, Any]:
        """Load a run or raise KeyError for HTTP 404 translation."""
        record = self.repository.get(run_id)
        if record is None:
            raise KeyError(run_id)
        return record

    def list(self) -> list[dict[str, Any]]:
        """List persisted runs newest first."""
        return self.repository.list()

    def evaluate(self, run_id: str) -> dict[str, Any]:
        """Calculate and persist hidden-label metrics."""
        record = self.require(run_id)
        result = evaluate_run(record, self.settings)
        self.repository.update(run_id, evaluation=result, updated_at=utc_now_iso())
        return result

    def events(self, run_id: str) -> list[dict[str, Any]]:
        """Return the safe append-only event projection."""
        return (self.require(run_id).get("state") or {}).get("events", [])


class RunService(RunExecutionMixin, RunQueryMixin):
    """Combine execution and query behavior with shared dependencies."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Create shared settings, repository, graph, and task registry."""
        self.settings = settings or get_settings()
        self.repository = RunRepository(self.settings)
        self.graph = build_orchestrator_graph(self.settings)
        self._tasks: dict[str, asyncio.Task] = {}


_service: RunService | None = None


def get_run_service() -> RunService:
    """Return one FastAPI service singleton per process."""
    global _service
    if _service is None:
        _service = RunService()
    return _service
