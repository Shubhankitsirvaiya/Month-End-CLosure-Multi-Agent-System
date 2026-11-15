"""Generate deterministic teaching data once before the test session."""
import os
import sys
from pathlib import Path

# The parent workspace contains another teaching project named ``app``. Put this
# repository first so pytest cannot accidentally import that sibling package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
# Unit tests construct the real OpenAI adapter but never make paid model calls.
# A dummy key prevents configuration failure while preserving production behavior.
# Assign directly because a developer may already have an empty OPENAI_API_KEY
# in the parent shell.  Tests construct the real adapter but do not invoke it.
os.environ["OPENAI_API_KEY"] = "test-key-not-used-for-network-calls"
os.environ["POLICY_RETRIEVAL_MODE"] = "local"
for module_name in list(sys.modules):
    if module_name == "app" or module_name.startswith("app."):
        module_file = str(getattr(sys.modules[module_name], "__file__", ""))
        if module_file and str(PROJECT_ROOT) not in module_file:
            del sys.modules[module_name]

import pytest
from scripts.generate_demo_data import main as generate_demo_data


@pytest.fixture(scope="session", autouse=True)
def generated_data():
    """Rebuild deterministic fixtures once so tests never depend on stale files."""
    generate_demo_data()
