from fastapi.testclient import TestClient
from app.api.main import app


def test_health_and_agent_registry():
    """Verify the service starts and publishes every approved specialist."""
    # Context manager runs FastAPI lifespan startup and shutdown hooks.
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        agents = client.get("/api/agents").json()
        assert {"reconciliation", "valuation", "accounting", "risk_policy", "evidence_critic"} <= set(agents)
