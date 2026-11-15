"""Retrieval layer for the fictional finance policy library.

The default TF-IDF index is intentionally local and deterministic. When the
OpenAI provider and API key are configured, the same chunks are embedded into a
persistent Chroma collection. Both modes return identical citation contracts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import Settings, get_settings
from app.core.models import EvidenceItem


@dataclass(slots=True)
class PolicyChunk:
    """One section-level retrieval unit with citation metadata.

    Section-level chunks preserve human-readable policy boundaries better than
    arbitrary character windows for this small, well-structured corpus.
    """
    chunk_id: str
    policy_id: str
    title: str
    section: str
    version: str
    effective_date: str
    source_file: str
    text: str


def _extract_metadata(text: str, filename: str) -> dict[str, str]:
    """Extract controlled metadata from a Markdown policy.

    Args:
        text: Complete Markdown document.
        filename: Used as a safe policy-ID fallback when metadata is missing.

    Returns:
        Policy ID, version, effective date, and title used in citations.
    """
    title = text.splitlines()[0].removeprefix("# ").strip()

    def value(label: str, fallback: str = "") -> str:
        """Read one bold Markdown metadata field such as ``**Version:**``."""
        match = re.search(rf"\*\*{re.escape(label)}:\*\*\s*(.+)", text)
        return match.group(1).strip() if match else fallback

    return {
        "policy_id": value("Policy ID", filename.split("_")[0]),
        "version": value("Version", "unknown"),
        "effective_date": value("Effective date", "unknown"),
        "title": title,
    }


def load_policy_chunks(policy_root: Path) -> list[PolicyChunk]:
    """Split every policy at ``##`` headings and retain source metadata.

    Args:
        policy_root: Directory containing governed Markdown policy files.

    Returns:
        Ordered chunks with stable IDs. Stable IDs are important because audit
        citations must not change merely because the process restarted.
    """
    # One shared list preserves deterministic file/section order.
    chunks: list[PolicyChunk] = []
    # Sorting guarantees stable chunk IDs across operating systems.
    for path in sorted(policy_root.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        metadata = _extract_metadata(text, path.name)
        # Text before the first ## heading belongs to an Overview section.
        current_section = "Overview"
        current_lines: list[str] = []

        def flush() -> None:
            """Persist the section accumulated since the previous heading."""
            if not current_lines:
                return
            body = "\n".join(current_lines).strip()
            if not body:
                return
            # Per-file numbering plus policy ID forms a stable, readable citation.
            chunk_number = sum(1 for item in chunks if item.source_file == path.name) + 1
            chunks.append(PolicyChunk(
                chunk_id=f"{metadata['policy_id']}-{chunk_number:02d}",
                policy_id=metadata["policy_id"],
                title=metadata["title"],
                section=current_section,
                version=metadata["version"],
                effective_date=metadata["effective_date"],
                source_file=path.name,
                text=body,
            ))

        # Skip the H1 title because it is stored separately as metadata.
        for line in text.splitlines()[1:]:
            if line.startswith("## "):
                flush()
                current_lines = []
                current_section = line.removeprefix("## ").strip()
            # Metadata lines should not dominate semantic retrieval results.
            elif not line.startswith("**Policy ID:") and not line.startswith("**Version:") and not line.startswith("**Effective date:") and not line.startswith("**Owner:"):
                current_lines.append(line)
        flush()
    return chunks


class PolicyRetriever:
    """Policy retrieval with metadata filtering and reproducible citations."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Build the local index and optionally initialize OpenAI + Chroma.

        ``ngram_range=(1, 2)`` indexes individual terms and two-word phrases;
        phrases such as "human approval" are valuable in policy retrieval.
        ``min_df=1`` is appropriate because the corpus is intentionally small.
        """
        self.settings = settings or get_settings()
        # Chunk once when the retriever is created, not for every search call.
        self.chunks = load_policy_chunks(self.settings.policy_root)
        if not self.chunks:
            raise RuntimeError(f"No policy documents found in {self.settings.policy_root}")
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        # fit_transform learns vocabulary and creates one sparse vector per chunk.
        self._matrix = self._vectorizer.fit_transform([chunk.text for chunk in self.chunks])
        self._chroma_collection = None
        self._embedding_model = None
        # Embedding mode is opt-in because it requires a secret and incurs API cost.
        if self.settings.llm_provider.lower() == "openai" and self.settings.openai_api_key:
            self._configure_chroma()

    def _configure_chroma(self) -> None:
        """Create or reuse a persistent embedding index.

        Only missing chunk IDs are embedded, which avoids unnecessary API cost
        and makes repeated application starts idempotent.
        """
        import chromadb
        from langchain_openai import OpenAIEmbeddings

        self._embedding_model = OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            api_key=self.settings.openai_api_key,
        )
        client = chromadb.PersistentClient(path=str(self.settings.runtime_root / "chroma"))
        # Cosine distance compares direction rather than vector magnitude and is
        # the standard similarity measure for normalized text embeddings.
        collection = client.get_or_create_collection("month_end_policies", metadata={"hnsw:space": "cosine"})
        # Read only IDs; documents/embeddings are unnecessary for this cache check.
        existing = set(collection.get(include=[]).get("ids", []))
        missing = [chunk for chunk in self.chunks if chunk.chunk_id not in existing]
        if missing:
            # Batch embedding is cheaper and faster than one request per section.
            embeddings = self._embedding_model.embed_documents([chunk.text for chunk in missing])
            collection.add(
                ids=[chunk.chunk_id for chunk in missing],
                embeddings=embeddings,
                documents=[chunk.text for chunk in missing],
                metadatas=[{
                    "policy_id": chunk.policy_id,
                    "title": chunk.title,
                    "section": chunk.section,
                    "version": chunk.version,
                    "effective_date": chunk.effective_date,
                    "source_file": chunk.source_file,
                } for chunk in missing],
            )
        self._chroma_collection = collection

    def search(
        self,
        query: str,
        *,
        top_k: int = 4,
        policy_ids: list[str] | None = None,
        portfolio_id: str | None = None,
        close_period: str | None = None,
    ) -> list[EvidenceItem]:
        """Return policy evidence with reproducible citations.

        Args:
            query: Investigation language converted into a retrieval vector.
            top_k: Maximum evidence chunks returned. A small value limits prompt
                noise and reviewer burden.
            policy_ids: Optional allow-list for domain-specific retrieval.
            portfolio_id: Copied into evidence for trace correlation; it does not
                alter policy text.
            close_period: Accounting period attached to the evidence record.

        Returns:
            Evidence ordered from strongest to weakest match. Scores measure
            retrieval similarity only; they are never used as approval authority.
        """

        # Filtering happens before ranking so an irrelevant policy cannot win
        # merely because it contains similar vocabulary.
        allowed = [
            index for index, chunk in enumerate(self.chunks)
            if not policy_ids or chunk.policy_id in set(policy_ids)
        ]
        if not allowed:
            return []

        # The Chroma branch currently handles unrestricted corpus searches. When
        # an allow-list is supplied we use the deterministic local index so the
        # policy boundary is guaranteed without relying on vector-store filters.
        if self._chroma_collection is not None and self._embedding_model is not None and not policy_ids:
            query_embedding = self._embedding_model.embed_query(query)
            result = self._chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, len(self.chunks)),
                include=["documents", "metadatas", "distances"],
            )
            evidence: list[EvidenceItem] = []
            for chunk_id, document, metadata, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            ):
                evidence.append(EvidenceItem(
                    evidence_id=chunk_id,
                    source_type="policy",
                    source_name=metadata["source_file"],
                    portfolio_id=portfolio_id,
                    close_period=close_period,
                    effective_date=metadata["effective_date"],
                    version=metadata["version"],
                    section=metadata["section"],
                    excerpt=document[:900],
                    score=max(0.0, 1.0 - float(distance)),
                    authoritative=True,
                ))
            return evidence

        # Transform uses the existing policy vocabulary; it does not refit per query.
        query_vector = self._vectorizer.transform([query])
        # Cosine similarity compares query direction with every chunk vector.
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        ranked = sorted(allowed, key=lambda idx: scores[idx], reverse=True)[:top_k]
        return [
            EvidenceItem(
                evidence_id=self.chunks[idx].chunk_id,
                source_type="policy",
                source_name=self.chunks[idx].source_file,
                portfolio_id=portfolio_id,
                close_period=close_period,
                effective_date=self.chunks[idx].effective_date,
                version=self.chunks[idx].version,
                section=self.chunks[idx].section,
                excerpt=self.chunks[idx].text[:900],
                score=round(float(scores[idx]), 4),
                authoritative=True,
            )
            for idx in ranked
        ]

