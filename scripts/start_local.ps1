# Run after activating the repository virtual environment. ``$PSScriptRoot`` is
# the directory containing this file, so moving one level up locates project root
# regardless of the caller's current working directory.
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
# Each long-running service receives its own hidden PowerShell process. ``-NoExit``
# keeps that process alive so the Python server continues serving requests.
#
# Start A2A first: the FastAPI orchestrator will discover/call it on port 9000.
Start-Process powershell -WindowStyle Hidden -WorkingDirectory $projectRoot -ArgumentList '-NoExit', '-Command', 'python -m uvicorn remote_agents.valuation_agent.app:app --port 9000'
# The main API owns validation, orchestration, checkpoints, approval, and metrics.
Start-Process powershell -WindowStyle Hidden -WorkingDirectory $projectRoot -ArgumentList '-NoExit', '-Command', 'python -m uvicorn app.api.main:app --port 8000'
# Streamlit is intentionally an API client and never imports finance backend logic.
Start-Process powershell -WindowStyle Hidden -WorkingDirectory $projectRoot -ArgumentList '-NoExit', '-Command', 'python -m streamlit run app/ui/streamlit_app.py --server.port 8501'
# Print discoverable endpoints so a learner can verify each boundary independently.
Write-Host 'Valuation A2A: http://localhost:9000/.well-known/agent-card.json'
Write-Host 'FastAPI docs:  http://localhost:8000/docs'
Write-Host 'Streamlit UI:  http://localhost:8501'
