from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

from .aggregation import AggregationEvaluation, evaluate_aggregation_expression, grouped_count
from .retrieval import BM25Index, EmbeddingClient, EmbeddingIndex, build_search_documents


SEARCH_RESULT_METADATA = {"query_terms", "matched_terms", "score_details", "rerank_score", "rerank_reason"}


SEARCH_STOPWORDS = {
    "and",
    "are",
    "call",
    "calls",
    "find",
    "for",
    "have",
    "mention",
    "mentions",
    "show",
    "the",
    "which",
    "with",
}


@dataclass
class ToolEvent:
    step: int
    tool: str
    arguments: dict[str, Any]
    result_summary: str
    latency_ms: float | None = None
    tokens: dict[str, Any] | None = None
    fallback_reason: str | None = None
    prompt_hash: str | None = None
    validation_result: str | None = None


class SilverCorpus:
    def __init__(
        self,
        *,
        manifest: dict[str, Any],
        catalog: dict[str, Any],
        records: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        field_candidates: list[dict[str, Any]] | None = None,
        schema_negotiation: dict[str, Any] | None = None,
        embedding_client: EmbeddingClient | None = None,
        embedding_cache_path: Path | None = None,
    ) -> None:
        self.manifest = manifest
        self.catalog = catalog
        self.records = records
        self.chunks = chunks
        self.field_candidates = field_candidates or []
        self.schema_negotiation = schema_negotiation or {}
        self.fields = {field["name"]: field for field in catalog.get("fields", [])}
        self.records_by_id = {record["call_id"]: record for record in records}
        self.chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        self.embedding_client = embedding_client
        self.embedding_cache_path = embedding_cache_path
        self._search_documents: list[dict[str, Any]] | None = None
        self._bm25_index: BM25Index | None = None
        self._embedding_index: EmbeddingIndex | None = None

    @classmethod
    def from_dir(
        cls,
        path: Path,
        *,
        embedding_client: EmbeddingClient | None = None,
        embedding_cache_path: Path | None = None,
    ) -> "SilverCorpus":
        return cls(
            manifest=read_json(path / "manifest.json"),
            catalog=read_json(path / "silver_schema_catalog.json"),
            records=read_jsonl(path / "silver_calls.jsonl"),
            chunks=read_jsonl(path / "chunks.jsonl"),
            field_candidates=read_json_optional(path / "field_candidates.json", default=[]),
            schema_negotiation=read_json_optional(path / "schema_negotiation.json", default={}),
            embedding_client=embedding_client,
            embedding_cache_path=embedding_cache_path,
        )

    def query_silver(self, filters: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
        matched = [record for record in self.records if record_matches(record, filters)]
        return matched if limit is None else matched[:limit]

    def aggregate_silver(self, group_by: str, filters: dict[str, Any], *, expression: str | None = None) -> dict[str, Any]:
        return self.aggregate_silver_result(group_by, filters, expression=expression).result

    def aggregate_silver_result(
        self,
        group_by: str,
        filters: dict[str, Any],
        *,
        expression: str | None = None,
    ) -> AggregationEvaluation:
        records = self.query_silver(filters)
        if expression:
            return evaluate_aggregation_expression(expression, records, self.fields)
        return AggregationEvaluation(result=grouped_count(records, group_by), used_fields=[group_by] if group_by else [])

    def search_chunks(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return self.lexical_overlap_search_chunks(query, filters=filters, limit=limit)

    def lexical_overlap_search_chunks(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        allowed_call_ids = None
        if filters:
            allowed_call_ids = {record["call_id"] for record in self.query_silver(filters)}
        terms = {
            term
            for term in re.findall(r"[a-z0-9_]+", query.lower())
            if len(term) > 2 and term not in SEARCH_STOPWORDS
        }
        scored: list[tuple[int, dict[str, Any]]] = []
        for chunk in self.chunks:
            if allowed_call_ids is not None and chunk["call_id"] not in allowed_call_ids:
                continue
            text = chunk["text"].lower()
            score = sum(1 for term in terms if term in text)
            if score:
                scored.append((score, chunk))
        return [chunk for _, chunk in sorted(scored, key=lambda item: (-item[0], item[1]["chunk_id"]))[:limit]]

    def bm25_search_chunks(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        allowed_call_ids = {record["call_id"] for record in self.query_silver(filters)} if filters else None
        scored = self.bm25_index().search(query, allowed_call_ids=allowed_call_ids, limit=limit)
        return self.chunks_for_scored_results(scored)

    def embedding_search_chunks(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if self.embedding_client is None:
            raise RuntimeError("Embedding retrieval requires an embedding client.")
        allowed_call_ids = {record["call_id"] for record in self.query_silver(filters)} if filters else None
        scored = self.embedding_index().search(query, allowed_call_ids=allowed_call_ids, limit=limit)
        return self.chunks_for_scored_results(scored)

    def hybrid_search_chunks(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        bm25 = self.bm25_search_chunks(query, filters=filters, limit=limit)
        embedding = self.embedding_search_chunks(query, filters=filters, limit=limit)
        return merge_chunks_by_rank([bm25, embedding], limit=limit)

    def fetch_chunks(self, refs_or_ids: list[str]) -> list[dict[str, Any]]:
        chunk_ids = [normalize_chunk_ref(ref) for ref in refs_or_ids]
        chunks = []
        for chunk_id in chunk_ids:
            chunk = self.chunks_by_id.get(chunk_id)
            if chunk:
                chunks.append(chunk)
        return chunks

    def bm25_backend_name(self) -> str:
        return self.bm25_index().backend_name

    def search_documents(self) -> list[dict[str, Any]]:
        if self._search_documents is None:
            self._search_documents = build_search_documents(
                catalog=self.catalog,
                records_by_id=self.records_by_id,
                chunks=self.chunks,
            )
        return self._search_documents

    def bm25_index(self) -> BM25Index:
        if self._bm25_index is None:
            self._bm25_index = BM25Index(self.search_documents())
        return self._bm25_index

    def embedding_index(self) -> EmbeddingIndex:
        if self.embedding_client is None:
            raise RuntimeError("Embedding retrieval requires an embedding client.")
        if self._embedding_index is None:
            self._embedding_index = EmbeddingIndex(
                self.search_documents(),
                client=self.embedding_client,
                cache_path=self.embedding_cache_path,
            )
        return self._embedding_index

    def chunks_for_scored_results(self, scored_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for result in scored_results:
            chunk = self.chunks_by_id.get(result["chunk_id"])
            if not chunk:
                continue
            enriched = dict(chunk)
            for key, value in result.items():
                if key.endswith("_score") or key in SEARCH_RESULT_METADATA:
                    enriched[key] = value
            chunks.append(enriched)
        return chunks


def record_matches(record: dict[str, Any], filters: dict[str, Any]) -> bool:
    fields = record.get("fields", {})
    for name, expected in filters.items():
        current = fields.get(name)
        if isinstance(current, list):
            if isinstance(expected, list):
                if not set(expected).intersection(current):
                    return False
            elif expected not in current:
                return False
        elif current != expected:
            return False
    return True


def normalize_chunk_ref(ref: str) -> str:
    return ref.removeprefix("chunk:")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_optional(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def merge_chunks_by_rank(chunk_lists: list[list[dict[str, Any]]], *, limit: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_length = max((len(chunks) for chunks in chunk_lists), default=0)
    for rank in range(max_length):
        for chunks in chunk_lists:
            if rank >= len(chunks):
                continue
            chunk = chunks[rank]
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id not in seen:
                merged.append(chunk)
                seen.add(chunk_id)
            if len(merged) >= limit:
                return merged
    return merged
