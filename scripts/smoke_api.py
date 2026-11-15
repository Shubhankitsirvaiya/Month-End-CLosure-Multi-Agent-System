"""Exercise the exact API lifecycle used by Streamlit."""
import time
from fastapi.testclient import TestClient
from app.api.main import app


def main() -> None:
    """Run the create -> poll -> approve -> resume API lifecycle."""
    with TestClient(app) as client:
        response = client.post("/api/runs", json={
            "scenario_id": "material_pricing_case",
            "portfolio_id": "MR-CREDIT-HY",
            "close_period": "2026-10-31",
            "user_goal": "Complete the controlled month-end review.",
        })
        response.raise_for_status()
        run_id = response.json()["run_id"]
        # Allow enough time for an offline A2A discovery timeout plus all agents.
        for _ in range(180):
            run = client.get(f"/api/runs/{run_id}").json()
            if run["status"] in {"approval_pending", "completed", "failed"}:
                break
            time.sleep(0.2)
        print({"run_id": run_id, "status": run["status"], "findings": len(run.get("state", {}).get("exceptions", []))})
        if run["status"] == "approval_pending":
            client.post(f"/api/runs/{run_id}/approval", json={
                "decision": "approved", "reviewer": "API smoke test", "comments": "Evidence reviewed.",
            }).raise_for_status()
            for _ in range(100):
                run = client.get(f"/api/runs/{run_id}").json()
                if run["status"] in {"completed", "failed", "rejected"}:
                    break
                time.sleep(0.2)
        print({"final_status": run["status"], "approval": run.get("state", {}).get("approval_decision")})
        if run["status"] != "completed":
            raise SystemExit(f"API smoke test did not complete: {run}")


if __name__ == "__main__":
    main()
