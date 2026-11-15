from fastapi.testclient import TestClient
from remote_agents.valuation_agent.app import app


def test_agent_card_declares_valuation_skill():
    """Ensure discovery tells orchestrators what the remote agent can do."""
    # TestClient exercises the HTTP boundary without opening a real network port.
    client = TestClient(app)
    response = client.get("/.well-known/agent-card.json")
    assert response.status_code == 200
    card = response.json()
    assert card["name"]
    assert any("valuation" in str(skill).lower() for skill in card["skills"])


def test_a2a_rejects_unknown_json_rpc_method():
    """Ensure unsupported methods return a protocol error, not silent success."""
    client = TestClient(app)
    # Deliberately send an unsupported method to verify JSON-RPC error handling.
    response = client.post("/", json={"jsonrpc": "2.0", "id": "1", "method": "unknown", "params": {}})
    assert response.status_code == 200
    assert response.json()["error"]["code"] == -32601
