"""Development smoke test for the complete supervisor graph."""
import asyncio
from app.agents.orchestrator_modular import build_orchestrator_graph
from langgraph.types import Command


async def main() -> None:
    """Invoke the graph, inspect its interrupt, and resume from checkpoint."""
    graph = build_orchestrator_graph()
    result = await graph.ainvoke(
        {
            "run_id": "smoke-001",
            "scenario_id": "material_breaks",
            "portfolio_id": "MR-CREDIT-HY",
            "close_period": "2026-10-31",
            "user_goal": "Complete close review",
        },
        config={"configurable": {"thread_id": "smoke-001"}},
    )
    print({
        "status": result.get("status"),
        "exceptions": len(result.get("exceptions", [])),
        "agents": sorted(set(result.get("completed_agents", []))),
        "interrupt": bool(result.get("__interrupt__")),
        "errors": result.get("errors", []),
        "summaries": [item.get("summary") for item in result.get("agent_outputs", [])],
    })
    if result.get("__interrupt__"):
        resumed = await graph.ainvoke(
            Command(resume={
                "decision": "approved",
                "reviewer": "Automated smoke-test reviewer",
                "comments": "The controlled test accepts the evidence package.",
            }),
            config={"configurable": {"thread_id": "smoke-001"}},
        )
        print({
            "resumed_status": resumed.get("status"),
            "final_exception_count": len((resumed.get("final_review_package") or {}).get("exceptions", [])),
        })


if __name__ == "__main__":
    asyncio.run(main())
