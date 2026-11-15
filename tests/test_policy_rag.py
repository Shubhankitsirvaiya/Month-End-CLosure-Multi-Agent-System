from app.core.config import get_settings
from app.rag.policy_rag_modular import PolicyRetriever


def test_policy_retriever_returns_citable_policy_sections():
    """Verify retrieval returns real source files and decision-relevant text."""
    retriever = PolicyRetriever(get_settings())
    # top_k=3 keeps the assertion focused on the strongest policy evidence.
    results = retriever.search("Who must approve a material stale price override?", top_k=3)
    assert results
    # Every result must be traceable to a real governed Markdown source.
    assert all(item.source_name.endswith(".md") for item in results)
    assert any("approval" in item.excerpt.lower() or "valuation" in item.excerpt.lower() for item in results)
