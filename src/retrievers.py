"""Retrieval engines for Day 7:
1. SparseRetriever (TF-IDF & BM25)
2. DenseRetriever (Qdrant Cloud + OpenAI 1536d)
3. HybridRetriever (Dense + Sparse via Reciprocal Rank Fusion - RRF)
4. TwoStageReranker (Cohere rerank-v3.5)
"""
import json
import math
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

# Load environment
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def tokenize(text: str) -> list[str]:
    """Normalize Unicode NFC, casefold, and extract alphanumeric tokens."""
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


class TfidfRetriever:
    """TF-IDF lexical retriever with metadata filter support."""

    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        counts = [Counter(tokenize(c['text'])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter) -> dict[str, float]:
        vector = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {term: v / norm for term, v in vector.items()}

    def search(self, query: str, top_k: int = 3, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        query_vec = self._vector(Counter(tokenize(query)))
        results = []
        for chunk, vector in zip(self.chunks, self.vectors):
            # Check metadata filter
            if filters:
                meta = chunk.get('metadata', {})
                match = all(meta.get(k) == v for k, v in filters.items())
                if not match:
                    continue

            score = sum(v * vector.get(t, 0.0) for t, v in query_vec.items())
            if score > 0:
                results.append({**chunk, 'score': score, 'strategy': 'tfidf'})

        results.sort(key=lambda c: (-c['score'], c['id']))
        for rank, r in enumerate(results[:top_k], start=1):
            r['rank'] = rank
        return results[:top_k]


class BM25Retriever:
    """BM25 lexical retriever (BM25Okapi) with metadata filter support."""

    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(c['text']) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 3, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)
        results = []

        for idx, score in enumerate(raw_scores):
            if score <= 0:
                continue
            chunk = self.chunks[idx]

            # Check metadata filter if present
            if filters:
                meta = chunk.get('metadata', {})
                match = all(meta.get(k) == v for k, v in filters.items())
                if not match:
                    continue

            item = dict(chunk)
            item['score'] = float(score)
            item['strategy'] = 'bm25'
            results.append(item)

        results.sort(key=lambda c: (-c['score'], c['id']))
        top_results = results[:top_k]
        for rank, r in enumerate(top_results, start=1):
            r['rank'] = rank
        return top_results


class DenseRetriever:
    """Dense semantic retriever using OpenAI text-embedding-3-small and Qdrant Cloud."""

    def __init__(self, collection_name: str = "strategy_b", dimensions: int = 1536):
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        if not url or not api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY must be set in .env")

        self.client = QdrantClient(url=url, api_key=api_key, timeout=15)
        self.openai_client = OpenAI(api_key=openai_key)
        self.collection_name = collection_name
        self.dimensions = dimensions

    def embed_query(self, query: str) -> list[float]:
        for attempt in range(1, 4):
            try:
                resp = self.openai_client.embeddings.create(
                    input=[query.replace("\n", " ")],
                    model="text-embedding-3-small",
                    timeout=30.0
                )
                return resp.data[0].embedding
            except Exception as e:
                if attempt < 3:
                    time.sleep(2 * attempt)
                else:
                    raise e

    def search(self, query: str, top_k: int = 3, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        q_vec = self.embed_query(query)

        query_filter = None
        if filters:
            conditions = []
            for k, v in filters.items():
                conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
            if conditions:
                query_filter = models.Filter(must=conditions)

        res = None
        for attempt in range(1, 4):
            try:
                res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=q_vec,
                    limit=top_k,
                    query_filter=query_filter,
                    with_payload=True
                )
                break
            except Exception as e:
                if attempt < 3:
                    time.sleep(2 * attempt)
                else:
                    raise e

        results = []
        for rank, pt in enumerate(res.points, start=1):
            payload = pt.payload or {}
            chunk_id = payload.get("chunk_id", str(pt.id))
            results.append({
                "id": chunk_id,
                "text": payload.get("text", ""),
                "source_id": payload.get("document_id", "chip-huyen-ai-engineering-ch1-2"),
                "source": payload.get("source", ""),
                "page": payload.get("page", 0),
                "metadata": {
                    "chapter": payload.get("chapter"),
                    "section": payload.get("section"),
                    "char_length": len(payload.get("text", ""))
                },
                "score": float(pt.score),
                "rank": rank,
                "strategy": "dense"
            })
        return results


class HybridRetriever:
    """Hybrid Retriever combining Dense (Qdrant) and Sparse (BM25) via Reciprocal Rank Fusion (RRF)."""

    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: BM25Retriever, k: int = 60):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.k = k  # RRF constant

    def search(self, query: str, top_k: int = 3, filters: dict[str, Any] | None = None,
               candidate_pool: int = 20) -> list[dict[str, Any]]:
        # 1. Fetch larger candidate pools from both retrievers
        pool_size = max(candidate_pool, top_k * 4)
        dense_results = self.dense_retriever.search(query, top_k=pool_size, filters=filters)
        sparse_results = self.sparse_retriever.search(query, top_k=pool_size, filters=filters)

        # 2. Build rank maps
        dense_rank_map = {item['id']: (rank, item) for rank, item in enumerate(dense_results, start=1)}
        sparse_rank_map = {item['id']: (rank, item) for rank, item in enumerate(sparse_results, start=1)}

        all_ids = set(dense_rank_map.keys()) | set(sparse_rank_map.keys())
        penalty_rank = pool_size + 10

        rrf_scored = []
        for cid in all_ids:
            dense_info = dense_rank_map.get(cid)
            sparse_info = sparse_rank_map.get(cid)

            d_rank = dense_info[0] if dense_info else penalty_rank
            s_rank = sparse_info[0] if sparse_info else penalty_rank

            # Reciprocal Rank Fusion formula
            rrf_score = (0.5 / (self.k + d_rank)) + (0.5 / (self.k + s_rank))

            # Pick base chunk object
            base_item = dict(dense_info[1] if dense_info else sparse_info[1])
            base_item['score'] = float(rrf_score)
            base_item['dense_rank'] = d_rank if dense_info else None
            base_item['sparse_rank'] = s_rank if sparse_info else None
            base_item['strategy'] = 'hybrid_rrf'
            rrf_scored.append(base_item)

        # 3. Sort by RRF score descending
        rrf_scored.sort(key=lambda c: (-c['score'], c['id']))
        top_results = rrf_scored[:top_k]
        for rank, r in enumerate(top_results, start=1):
            r['rank'] = rank
        return top_results


class CohereReranker:
    """Two-Stage Cross-Encoder Reranker using Cohere API v1 (model: rerank-v3.5)."""

    def __init__(self, model: str = "rerank-v3.5", api_key: str | None = None):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY must be set in .env")
        self.model = model
        self.api_url = "https://api.cohere.ai/v1/rerank"

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
        if not candidates:
            return []

        doc_texts = [c.get("text", " ") for c in candidates]
        payload = {
            "model": self.model,
            "query": query.strip(),
            "documents": doc_texts,
            "top_n": min(top_k, len(candidates))
        }

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )

        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    results = data.get("results", [])
                    reranked = []
                    for rank, res_item in enumerate(results, start=1):
                        orig_idx = res_item["index"]
                        chunk = dict(candidates[orig_idx])
                        chunk["original_rank"] = orig_idx + 1
                        chunk["score"] = float(res_item["relevance_score"])
                        chunk["rank"] = rank
                        chunk["strategy"] = "cohere_rerank"
                        reranked.append(chunk)
                    return reranked
            except Exception as e:
                if attempt < 3:
                    time.sleep(1.5 * attempt)
                else:
                    print(f"[CohereReranker] Warning: fallback to original candidates due to {e}")
                    return candidates[:top_k]
        return candidates[:top_k]
