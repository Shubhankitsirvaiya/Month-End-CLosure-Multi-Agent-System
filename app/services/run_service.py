"""Execute and resume the LangGraph workflow behind a stable service API."""
from __future__ import annotations

import asyncio
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

from langgraph.types import Command

from app.agents.orchestrator_modular import build_orchestrator_graph
from app.core.config import Settings, get_settings
from app.core.models import ApprovalDecisionRequest, RunCreateRequest, utc_now_iso
from app.evaluation.evaluator import evaluate_run
from app.storage.repository import RunRepository


def _jsonable(value: Any) -> Any:
    """Recursively convert Pydantic and tuple values into JSON-safe objects."""
    # Pydantic v2 models expose model_dump; convert before writing JSON.
    if hasattr(value, "model_dump"):
        return value.model_dump()
    # Recurse because LangGraph results may contain nested models and tuples.
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class RunService:
    """Own one compiled graph and expose non-blocking run operations."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Create one graph/checkpointer and one repository per API process."""
        self.settings = settings or get_settings()
        self.repository = RunRepository(self.settings)
        self.graph = build_orchestrator_graph(self.settings)
        self._tasks: dict[str, asyncio.Task] = {}

    def create(self, request: RunCreateRequest) -> dict[str, Any]:
        """Persist a queued run and schedule graph execution without blocking.

        Args:
            request: Already validated FastAPI request model.

        Returns:
            The initial record immediately; the background task updates it.
        """
        # Random suffix prevents collision when multiple users start the same scenario.
        run_id = f"close-{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        record = {
            "run_id": run_id,
            "scenario_id": request.scenario_id,
            "portfolio_id": request.portfolio_id,
            "close_period": request.close_period,
            "user_goal": request.user_goal,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "state": {},
            "evaluation": None,
            "error": None,
        }
        # Save before scheduling so the UI can immediately poll a known run ID.
        self.repository.save(record)
        # create_task schedules work on FastAPI's event loop and returns immediately.
        self._tasks[run_id] = asyncio.create_task(self._run_initial(record))
        return record

    async def _run_initial(self, record: dict[str, Any]) -> None:
        """Translate the persisted API record into initial graph state."""
        initial_state = {
            "run_id": record["run_id"], "scenario_id": record["scenario_id"],
            "portfolio_id": record["portfolio_id"], "close_period": record["close_period"],
            "user_goal": record["user_goal"],
        }
        await self._invoke(record["run_id"], initial_state)

    async def _invoke(self, run_id: str, graph_input: Any) -> None:
        """Invoke or resume the graph and project its state into persistence.

        ``graph_input`` is either initial state or ``Command(resume=...)``. Both
        use the same thread ID so LangGraph can locate the correct checkpoint.
        """
        try:
            self.repository.update(run_id, status="running", updated_at=utc_now_iso())
            # thread_id is the checkpoint lookup key used again during resume.
            config = {"configurable": {"thread_id": run_id}}
            result = _jsonable(await self.graph.ainvoke(graph_input, config=config))
            # LangGraph reports pending interrupts in a reserved result field.
            interrupts = result.pop("__interrupt__", []) if isinstance(result, dict) else []
            if interrupts:
                # This graph has one approval gate, so the first interrupt is the
                # complete operator packet exposed through FastAPI.
                first = interrupts[0]
                approval = _jsonable(getattr(first, "value", first))
                result["approval_request"] = approval
                result["approval_required"] = True
                result["status"] = "approval_pending"
            status = result.get("status", "completed")
            self.repository.update(run_id, status=status, state=result, updated_at=utc_now_iso())
        except Exception as exc:
            # Persist failure details because the background task has no caller
            # waiting to receive a normal exception response.
            self.repository.update(
                run_id, status="failed", updated_at=utc_now_iso(),
                error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)},
            )

    def resume(self, run_id: str, decision: ApprovalDecisionRequest) -> dict[str, Any]:
        """Schedule continuation of a run currently paused for approval."""
        record = self.require(run_id)
        # Reject duplicate or out-of-order approvals instead of resuming arbitrary state.
        if record["status"] != "approval_pending":
            raise ValueError("This run is not waiting for approval")
        payload = decision.model_dump()
        payload["decided_at"] = utc_now_iso()
        # Command(resume=payload) injects the decision at the interrupted node.
        self._tasks[run_id] = asyncio.create_task(self._invoke(run_id, Command(resume=payload)))
        return self.repository.update(run_id, status="resuming", updated_at=utc_now_iso())

    def require(self, run_id: str) -> dict[str, Any]:
        """Load a run or raise ``KeyError`` for API translation to HTTP 404."""
        record = self.repository.get(run_id)
        if record is None:
            raise KeyError(run_id)
        return record

    def list(self) -> list[dict[str, Any]]:
        """List persisted operator records."""
        return self.repository.list()

    def evaluate(self, run_id: str) -> dict[str, Any]:
        """Calculate and persist hidden-label outcome metrics."""
        record = self.require(run_id)
        result = evaluate_run(record, self.settings)
        self.repository.update(run_id, evaluation=result, updated_at=utc_now_iso())
        return result

    def events(self, run_id: str) -> list[dict[str, Any]]:
        """Return the safe event stream stored in shared state."""
        return (self.require(run_id).get("state") or {}).get("events", [])


_service: RunService | None = None


def get_run_service() -> RunService:
    """FastAPI dependency returning one service singleton per process."""
    global _service
    if _service is None:
        _service = RunService()
    return _service
