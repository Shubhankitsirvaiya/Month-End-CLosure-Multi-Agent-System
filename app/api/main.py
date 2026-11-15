"""Operations-oriented FastAPI boundary for the multi-agent workflow."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.agents.registry import get_agent_registry
from app.core.config import Settings, get_settings
from app.core.models import ApprovalDecisionRequest, RunCreateRequest
from app.data.loader import load_dataset_profile, load_scenarios
from app.services.run_service_modular import RunService, get_run_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate startup prerequisites before accepting traffic.

    FastAPI runs code before ``yield`` at startup and code after it at shutdown.
    Failing here is preferable to accepting a run that later lacks source data.
    """
    settings = get_settings()
    required = settings.data_root / "generated" / "portfolios.csv"
    if not required.exists():
        raise RuntimeError("Generated dataset is missing. Run: python -m scripts.generate_demo_data")
    yield


# Metadata appears automatically in /docs and tells clients what this service owns.
app = FastAPI(
    title="Month-End Portfolio Review Copilot API",
    version="1.0.0",
    description="Dynamic LangGraph supervision, specialist agents, A2A valuation, HITL approval, and hidden-label evaluation.",
    lifespan=lifespan,
)
# CORS allows expected local Streamlit origins instead of every website.
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict:
    """Report process health and non-secret runtime choices."""
    return {"status": "ok", "llm_provider": "openai", "a2a_valuation_url": settings.valuation_agent_url}


@app.get("/api/scenarios")
def scenarios() -> list[dict]:
    """Return curated business scenarios used to start meaningful demos."""
    return load_scenarios()


@app.get("/api/dataset/profile")
def dataset_profile() -> dict:
    """Return safe dataset counts and valid portfolio identifiers."""
    return load_dataset_profile()


@app.get("/api/agents")
def agents() -> dict[str, dict]:
    """Expose registered agent descriptions for UI transparency."""
    return get_agent_registry()


@app.get("/api/agents/valuation/card")
async def valuation_card(settings: Settings = Depends(get_settings)) -> dict:
    """Proxy remote Agent Card discovery without failing the whole UI.

    A four-second timeout is intentionally shorter than task execution timeout:
    discovery should be cheap, and operators need fast availability feedback.
    """
    url = settings.valuation_agent_url.rstrip("/") + "/.well-known/agent-card.json"
    try:
        # Async discovery does not block unrelated API requests.
        async with httpx.AsyncClient(timeout=4) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        return {"available": False, "url": url, "error": str(exc), "fallback": "local valuation LangGraph agent"}


@app.post("/api/runs", status_code=202)
async def create_run(request: RunCreateRequest, service: RunService = Depends(get_run_service)) -> dict:
    """Validate and enqueue a review, returning HTTP 202 without waiting."""
    # This endpoint stays async because RunService schedules the graph on the
    # current event loop and immediately returns a 202 response to the UI.
    return service.create(request)


@app.get("/api/runs")
def list_runs(service: RunService = Depends(get_run_service)) -> list[dict]:
    """List newest-first operator projections, not internal checkpoints."""
    return service.list()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, service: RunService = Depends(get_run_service)) -> dict:
    """Return one run, including its safe shared-state projection."""
    try:
        return service.require(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")


@app.get("/api/runs/{run_id}/events")
def get_events(run_id: str, service: RunService = Depends(get_run_service)) -> list[dict]:
    """Return audit events separately for lightweight polling clients."""
    try:
        return service.events(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")


@app.get("/api/runs/{run_id}/events/stream")
async def stream_events(run_id: str, service: RunService = Depends(get_run_service)):
    """Stream newly appended audit events using Server-Sent Events.

    SSE is one-way server-to-browser communication, ideal for progress updates.
    The generator stops at a terminal or human-waiting state so connections do
    not remain open indefinitely.
    """
    async def generator():
        """Yield only events not already sent, then terminate at a wait/end state."""
        # ``sent`` is an index into the append-only event list.
        sent = 0
        while True:
            try:
                record = service.require(run_id)
            except KeyError:
                yield f"event: error\ndata: {json.dumps({'message': 'Run not found'})}\n\n"
                return
            # Runs may exist before graph state is first persisted, hence ``or {}``.
            events = (record.get("state") or {}).get("events", [])
            for event in events[sent:]:
                yield f"id: {event.get('event_id', '')}\nevent: agent_event\ndata: {json.dumps(event)}\n\n"
            sent = len(events)
            if record.get("status") in {"completed", "failed", "rejected", "approval_pending"}:
                yield f"event: status\ndata: {json.dumps({'status': record.get('status')})}\n\n"
                return
            # Async sleep avoids CPU-heavy polling and yields to other requests.
            await asyncio.sleep(0.75)
    return StreamingResponse(generator(), media_type="text/event-stream")


@app.post("/api/runs/{run_id}/approval", status_code=202)
async def approve(run_id: str, decision: ApprovalDecisionRequest, service: RunService = Depends(get_run_service)) -> dict:
    """Resume a checkpointed graph with an accountable reviewer decision."""
    try:
        return service.resume(run_id, decision)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/runs/{run_id}/evaluate")
def evaluate(run_id: str, service: RunService = Depends(get_run_service)) -> dict:
    """Evaluate completed findings against labels hidden from every agent."""
    try:
        return service.evaluate(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")
