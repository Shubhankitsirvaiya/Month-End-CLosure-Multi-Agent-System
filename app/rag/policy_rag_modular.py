"""Modular policy retrieval used by the backend and step-by-step notebook."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import Settings, get_settings
from app.core.models import EvidenceItem


@dataclass(slots=True)
class PolicyChunk:
    """One section-level retrieval unit with complete citation metadata."""

    chunk_id: str
    policy_id: str
    title: str
    section: str
    version: str
    effective_date: str
    source_file: str
    text: str


def extract_policy_metadata(text: str, filename: str) -> dict[str, str]:
    """Read title, ID, version, and effective date from Markdown metadata."""
    title = text.splitlines()[0].removeprefix("# ").strip()

    def value(label: str, fallback: str = "") -> str:
        """Read one bold metadata field, returning a controlled fallback."""
        match = re.search(rf"\*\*{re.escape(label)}:\*\*\s*(.+)", text)
        return match.group(1).strip() if match else fallback

    return {
        "policy_id": value("Policy ID", filename.split("_")[0]),
        "version": value("Version", "unknown"),
        "effective_date": value("Effective date", "unknown"),
        "title": title,
    }


def load_policy_chunks(policy_root: Path) -> list[PolicyChunk]:
    """Split sorted Markdown policies at H2 headings into stable chunks."""
    chunks: list[PolicyChunk] = []
    for path in sorted(policy_root.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        metadata = extract_policy_metadata(text, path.name)
        current_section = "Overview"
        current_lines: list[str] = []

        def flush() -> None:
            """Store the current non-empty section before moving to the next."""
            if not current_lines:
                return
            body = "\n".join(current_lines).strip()
            if not body:
                return
            number = sum(item.source_file == path.name for item in chunks) + 1
            chunks.append(PolicyChunk(
                chunk_id=f"{metadata['policy_id']}-{number:02d}",
                policy_id=metadata["policy_id"], title=metadata["title"],
                section=current_section, version=metadata["version"],
                effective_date=metadata["effective_date"],
                source_file=path.name, text=body,
            ))

        for line in text.splitlines()[1:]:
            if line.startswith("## "):
                flush()
                current_lines = []
                current_section = line.removeprefix("## ").strip()
            elif not line.startswith(("**Policy ID:", "**Version:", "**Effective date:", "**Owner:")):
                current_lines.append(line)
        flush()
    return chunks


class ChromaIndexMixin:
    """Optional OpenAI-embedding and persistent-Chroma setup."""

    def _configure_chroma(self) -> None:
        """Embed only new chunk IDs and reuse the persistent collection."""
        import chromadb
        from langchain_openai import OpenAIEmbeddings

        self._embedding_model = OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            api_key=self.settings.openai_api_key,
        )
        client = chromadb.PersistentClient(path=str(self.settings.runtime_root / "chroma"))
        collection = client.get_or_create_collection(
            "month_end_policies", metadata={"hnsw:space": "cosine"}
        )
        existing = set(collection.get(include=[]).get("ids", []))
        missing = [chunk for chunk in self.chunks if chunk.chunk_id not in existing]
        if missing:
            # One batch reduces network overhead and embedding cost.
            embeddings = self._embedding_model.embed_documents([chunk.text for chunk in missing])
            collection.add(
                ids=[chunk.chunk_id for chunk in missing],
                embeddings=embeddings,
                documents=[chunk.text for chunk in missing],
                metadatas=[{
                    "policy_id": chunk.policy_id, "title": chunk.title,
                    "section": chunk.section, "version": chunk.version,
                    "effective_date": chunk.effective_date,
                    "source_file": chunk.source_file,
                } for chunk in missing],
            )
        self._chroma_collection = collection


class PolicySearchMixin:
    """Query-time retrieval shared by local TF-IDF and Chroma modes."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 4,
        policy_ids: list[str] | None = None,
        portfolio_id: str | None = None,
        close_period: str | None = None,
    ) -> list[EvidenceItem]:
        """Return ranked evidence after optional policy allow-list filtering."""
        allowed = [
            index for index, chunk in enumerate(self.chunks)
            if not policy_ids or chunk.policy_id in set(policy_ids)
        ]
        if not allowed:
            return []

        if self._chroma_collection is not None and self._embedding_model is not None and not policy_ids:
            result = self._chroma_collection.query(
                query_embeddings=[self._embedding_model.embed_query(query)],
                n_results=min(top_k, len(self.chunks)),
                include=["documents", "metadatas", "distances"],
            )
            evidence = []
            for chunk_id, document, metadata, distance in zip(
                result["ids"][0], result["documents"][0],
                result["metadatas"][0], result["distances"][0], strict=True,
            ):
                evidence.append(EvidenceItem(
                    evidence_id=chunk_id, source_type="policy",
                    source_name=metadata["source_file"], portfolio_id=portfolio_id,
                    close_period=close_period, effective_date=metadata["effective_date"],
                    version=metadata["version"], section=metadata["section"],
                    excerpt=document[:900], score=max(0.0, 1.0 - float(distance)),
                    authoritative=True,
                ))
            return evidence

        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        ranked = sorted(allowed, key=lambda index: scores[index], reverse=True)[:top_k]
        return [EvidenceItem(
            evidence_id=self.chunks[index].chunk_id, source_type="policy",
            source_name=self.chunks[index].source_file, portfolio_id=portfolio_id,
            close_period=close_period, effective_date=self.chunks[index].effective_date,
            version=self.chunks[index].version, section=self.chunks[index].section,
            excerpt=self.chunks[index].text[:900], score=round(float(scores[index]), 4),
            authoritative=True,
        ) for index in ranked]


class PolicyRetriever(ChromaIndexMixin, PolicySearchMixin):
    """Initialize policy chunks and the selected vector representation."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Load chunks and initialize local or embedding-backed retrieval."""
        # Settings controls corpus path and whether paid embedding mode is enabled.
        self.settings = settings or get_settings()
        self.chunks = load_policy_chunks(self.settings.policy_root)
        if not self.chunks:
            raise RuntimeError(f"No policy documents found in {self.settings.policy_root}")
        # Unigrams + bigrams capture terms and policy phrases in offline mode.
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self._matrix = self._vectorizer.fit_transform([chunk.text for chunk in self.chunks])
        self._chroma_collection = None
        self._embedding_model = None
        if self.settings.policy_retrieval_mode.lower() == "openai" and self.settings.openai_api_key:
            self._configure_chroma()
