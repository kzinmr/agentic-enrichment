from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Protocol

from .llm import DEFAULT_OPENAI_BASE_URL, LLMError, post_json


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "call",
    "calls",
    "find",
    "for",
    "from",
    "have",
    "how",
    "in",
    "is",
    "it",
    "mention",
    "mentions",
    "of",
    "on",
    "or",
    "show",
    "that",
    "the",
    "this",
    "to",
    "was",
    "we",
    "what",
    "which",
    "with",
}


class EmbeddingClient(Protocol):
    model: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbeddingClient:
    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key: str | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.timeout_seconds = timeout_seconds
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is required for embedding retrieval.")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = post_json(
            f"{self.base_url}/embeddings",
            {"model": self.model, "input": texts},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout_seconds=self.timeout_seconds,
        )
        data = response.get("data")
        if not isinstance(data, list):
            raise LLMError(f"Embedding response missing data array: {response}")
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        embeddings: list[list[float]] = []
        for item in ordered:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list):
                raise LLMError(f"Embedding response item missing embedding: {item}")
            embeddings.append([float(value) for value in embedding])
        if len(embeddings) != len(texts):
            raise LLMError(f"Embedding response count mismatch: expected {len(texts)}, got {len(embeddings)}")
        return embeddings


class BM25Index:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self._bm25s = None
        self._numpy = None
        self._fallback = FallbackBM25Index(documents)
        self._init_bm25s()

    def _init_bm25s(self) -> None:
        try:
            import bm25s
            import numpy as np
        except ImportError:
            return
        if not self.documents:
            return
        tokenized_documents = [english_tokenize(document["bm25_text"]) for document in self.documents]
        if not any(tokenized_documents):
            return
        retriever = bm25s.BM25(corpus=self.documents, backend="numpy", csc_backend="numpy")
        retriever.index(
            tokenized_documents,
            show_progress=False,
            leave_progress=False,
        )
        self._bm25s = retriever
        self._numpy = np

    @property
    def backend_name(self) -> str:
        return "bm25s" if self._bm25s is not None else "fallback_bm25"

    def search(self, query: str, *, allowed_call_ids: set[str] | None, limit: int) -> list[dict[str, Any]]:
        query_tokens = english_tokenize(query)
        if not query_tokens:
            return []
        if self._bm25s is None:
            return self._fallback.search(query, allowed_call_ids=allowed_call_ids, limit=limit)
        weight_mask = None
        candidate_count = len(self.documents)
        if allowed_call_ids is not None:
            allowed = [document["call_id"] in allowed_call_ids for document in self.documents]
            candidate_count = sum(1 for is_allowed in allowed if is_allowed)
            if candidate_count == 0:
                return []
            weight_mask = self._numpy.array(allowed, dtype="float32") if self._numpy is not None else None
        result = self._bm25s.retrieve(
            [query_tokens],
            corpus=self.documents,
            k=min(max(limit, 1), candidate_count),
            return_as="tuple",
            show_progress=False,
            leave_progress=False,
            weight_mask=weight_mask,
        )
        results: list[tuple[float, dict[str, Any]]] = []
        for document, score in zip(result.documents[0], result.scores[0], strict=True):
            score_value = float(score)
            if score_value <= 0:
                continue
            if allowed_call_ids is not None and document["call_id"] not in allowed_call_ids:
                continue
            results.append((score_value, document))
        return ranked_documents(
            results,
            limit=limit,
            score_key="bm25_score",
            details_by_chunk_id=self._fallback.explain_results(query_tokens, results, backend=self.backend_name),
        )


class FallbackBM25Index:
    def __init__(self, documents: list[dict[str, Any]], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [english_tokenize(document["bm25_text"]) for document in documents]
        self.doc_index_by_chunk_id = {document["chunk_id"]: index for index, document in enumerate(documents)}
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.doc_freqs: dict[str, int] = {}
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

    def search(self, query: str, *, allowed_call_ids: set[str] | None, limit: int) -> list[dict[str, Any]]:
        query_tokens = english_tokenize(query)
        results: list[tuple[float, dict[str, Any]]] = []
        for index, document in enumerate(self.documents):
            if allowed_call_ids is not None and document["call_id"] not in allowed_call_ids:
                continue
            score = self.score(query_tokens, index)
            if score > 0:
                results.append((score, document))
        return ranked_documents(
            results,
            limit=limit,
            score_key="bm25_score",
            details_by_chunk_id=self.explain_results(query_tokens, results, backend="fallback_bm25"),
        )

    def score(self, query_tokens: list[str], doc_index: int) -> float:
        return sum(detail["contribution"] for detail in self.term_details(query_tokens, doc_index))

    def explain_results(
        self,
        query_tokens: list[str],
        results: list[tuple[float, dict[str, Any]]],
        *,
        backend: str,
    ) -> dict[str, dict[str, Any]]:
        query_terms = dedupe(query_tokens)
        details_by_chunk_id: dict[str, dict[str, Any]] = {}
        for backend_score, document in results:
            doc_index = self.doc_index_by_chunk_id.get(document["chunk_id"])
            if doc_index is None:
                continue
            term_details = self.term_details(query_terms, doc_index)
            matched_terms = [detail["term"] for detail in term_details if detail["tf"] > 0]
            details_by_chunk_id[document["chunk_id"]] = {
                "query_terms": query_terms,
                "matched_terms": matched_terms,
                "score_details": {
                    "backend": backend,
                    "backend_score": round(float(backend_score), 6),
                    "explain_score": round(sum(detail["contribution"] for detail in term_details), 6),
                    "missing_terms": [term for term in query_terms if term not in matched_terms],
                    "terms": term_details,
                },
            }
        return details_by_chunk_id

    def term_details(self, query_tokens: list[str], doc_index: int) -> list[dict[str, Any]]:
        tokens = self.doc_tokens[doc_index]
        if not tokens:
            return []
        term_freqs: dict[str, int] = {}
        for token in tokens:
            term_freqs[token] = term_freqs.get(token, 0) + 1
        total_docs = len(self.documents)
        doc_length = self.doc_lengths[doc_index]
        details: list[dict[str, Any]] = []
        for token in dedupe(query_tokens):
            tf = term_freqs.get(token, 0)
            df = self.doc_freqs.get(token, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            contribution = 0.0
            if tf > 0:
                denom = tf + self.k1 * (1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1.0))
                contribution = idf * (tf * (self.k1 + 1)) / denom
            details.append(
                {
                    "term": token,
                    "tf": tf,
                    "df": df,
                    "idf": round(idf, 6),
                    "contribution": round(contribution, 6),
                }
            )
        return details


class EmbeddingIndex:
    def __init__(
        self,
        documents: list[dict[str, Any]],
        *,
        client: EmbeddingClient,
        cache_path: Path | None = None,
    ) -> None:
        self.documents = documents
        self.client = client
        self.cache_path = cache_path
        self.document_hash = documents_hash(documents, model=client.model)
        self._embeddings: list[list[float]] | None = None

    def search(self, query: str, *, allowed_call_ids: set[str] | None, limit: int) -> list[dict[str, Any]]:
        embeddings = self.document_embeddings()
        query_embedding = self.client.embed_texts([query])[0]
        results: list[tuple[float, dict[str, Any]]] = []
        for embedding, document in zip(embeddings, self.documents, strict=True):
            if allowed_call_ids is not None and document["call_id"] not in allowed_call_ids:
                continue
            score = cosine_similarity(query_embedding, embedding)
            results.append((score, document))
        return ranked_documents(results, limit=limit, score_key="embedding_score")

    def document_embeddings(self) -> list[list[float]]:
        if self._embeddings is not None:
            return self._embeddings
        cached = self.load_cache()
        if cached is not None:
            self._embeddings = cached
            return cached
        embeddings = self.client.embed_texts([document["embedding_text"] for document in self.documents])
        self._embeddings = embeddings
        self.write_cache(embeddings)
        return embeddings

    def load_cache(self) -> list[list[float]] | None:
        if self.cache_path is None or not self.cache_path.exists():
            return None
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        if payload.get("model") != self.client.model or payload.get("document_hash") != self.document_hash:
            return None
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(self.documents):
            return None
        return [[float(value) for value in embedding] for embedding in embeddings]

    def write_cache(self, embeddings: list[list[float]]) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": self.client.model, "document_hash": self.document_hash, "embeddings": embeddings}
        self.cache_path.write_text(json.dumps(payload), encoding="utf-8")


def build_search_documents(
    *,
    catalog: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    field_names = [field["name"] for field in catalog.get("fields", []) if field.get("search", {}).get("filterable", True)]
    documents: list[dict[str, Any]] = []
    for chunk in chunks:
        record = records_by_id.get(chunk["call_id"], {})
        field_text = silver_field_text(record, field_names)
        account_text = " ".join(str(record.get(name, "")) for name in ("account_name", "customer_id"))
        chunk_text = str(chunk.get("text", ""))
        documents.append(
            {
                "chunk_id": chunk["chunk_id"],
                "call_id": chunk["call_id"],
                "bm25_text": compose_bm25_text(chunk_text=chunk_text, field_text=field_text, account_text=account_text),
                "embedding_text": compose_embedding_text(
                    chunk_text=chunk_text,
                    field_text=field_text,
                    account_text=account_text,
                ),
            }
        )
    return documents


def compose_bm25_text(*, chunk_text: str, field_text: str, account_text: str) -> str:
    # BM25F-lite: one global index keeps IDF corpus-wide while light repetition approximates field weights.
    return "\n".join([chunk_text, chunk_text, field_text, account_text])


def compose_embedding_text(*, chunk_text: str, field_text: str, account_text: str) -> str:
    return f"account: {account_text}\nsilver_fields: {field_text}\ntranscript_chunk: {chunk_text}"


def silver_field_text(record: dict[str, Any], field_names: list[str]) -> str:
    fields = record.get("fields", {})
    parts: list[str] = []
    for name in field_names:
        value = fields.get(name)
        if isinstance(value, list):
            parts.extend(str(item).replace("_", " ") for item in value)
        elif value not in (None, "", False, "not_mentioned"):
            parts.append(str(value).replace("_", " "))
    return " ".join(parts)


def english_tokenize(value: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(value.lower().replace("_", " ")):
        token = match.group(0)
        if token.endswith("'s"):
            token = token[:-2]
        if len(token) > 1 and token not in ENGLISH_STOPWORDS:
            tokens.append(token)
    return tokens


def ranked_documents(
    scored_documents: list[tuple[float, dict[str, Any]]],
    *,
    limit: int,
    score_key: str,
    details_by_chunk_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    ranked = sorted(scored_documents, key=lambda item: (-item[0], item[1]["chunk_id"]))[:limit]
    results: list[dict[str, Any]] = []
    for score, document in ranked:
        result = {"chunk_id": document["chunk_id"], "call_id": document["call_id"], score_key: round(score, 6)}
        if details_by_chunk_id and document["chunk_id"] in details_by_chunk_id:
            result.update(details_by_chunk_id[document["chunk_id"]])
        results.append(result)
    return results


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def documents_hash(documents: list[dict[str, Any]], *, model: str) -> str:
    payload = {
        "model": model,
        "documents": [(document["chunk_id"], document["embedding_text"]) for document in documents],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
