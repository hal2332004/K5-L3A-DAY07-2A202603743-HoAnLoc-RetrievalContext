"""Benchmark and compare all retrieval strategies on the Dev split.
Evaluates:
- Sparse (TF-IDF)
- Sparse (BM25)
- Dense (Qdrant Cloud + OpenAI 1536d)
- Hybrid (Dense + BM25 via Reciprocal Rank Fusion)
- Two-Stage (Hybrid + Cohere Rerank-v3.5)
"""
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.retrievers import BM25Retriever, CohereReranker, DenseRetriever, HybridRetriever, TfidfRetriever
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    """Check whether a chunk contains any of the gold evidence on the correct page."""
    return any(
        chunk.get('source_id') == g['source_id']
        and chunk.get('page') == g['page']
        and g['evidence'] in chunk.get('text', '')
        for g in gold
    )


def pack_context(results: list[dict[str, Any]], budget: int = 800) -> tuple[str, list[str]]:
    """Pack chunks into context string within character budget and deduplicate IDs."""
    blocks, ids, seen = [], [], set()
    for c in results:
        cid = c.get('id', c.get('chunk_id'))
        if cid in seen:
            continue
        seen.add(cid)
        block = f"[{cid} | {c.get('source')} | page {c.get('page')}]\n{c.get('text')}"
        candidate_context = '\n\n'.join(blocks + [block])
        if len(candidate_context) <= budget:
            blocks.append(block)
            ids.append(cid)
    return '\n\n'.join(blocks), ids


def evaluate_engine(name: str, search_fn, questions: list[dict[str, Any]], budget: int = 800) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(f"\n---> Evaluating {name} on {len(questions)} Dev queries...")
    traces = []
    latencies = []

    for q in questions:
        q_id = q["id"]
        query_text = q["question"]
        gold = q["gold"]
        filters = q.get("filter")

        start = time.perf_counter()
        results = search_fn(query_text, top_k=3, filters=filters)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        context, context_ids = pack_context(results, budget=budget)

        marked_results = []
        for rank, c in enumerate(results, start=1):
            is_hit = hit(c, gold)
            item = dict(c)
            item['rank'] = rank
            item['hit'] = is_hit
            marked_results.append(item)

        hit1 = bool(marked_results and marked_results[0]['hit'])
        hit3 = any(c['hit'] for c in marked_results[:3])
        context_hit = any(c['hit'] and c['id'] in context_ids for c in marked_results[:3])

        traces.append({
            "query_id": q_id,
            "split": q["split"],
            "question": query_text,
            "strategy": name,
            "gold": gold,
            "filter": filters,
            "top_k": 3,
            "budget_chars": budget,
            "elapsed_ms": elapsed_ms,
            "hit1": hit1,
            "hit3": hit3,
            "context_hit": context_hit,
            "context_ids": context_ids,
            "context_chars": len(context),
            "results": marked_results
        })

    n = len(questions)
    hit1_count = sum(t['hit1'] for t in traces)
    hit3_count = sum(t['hit3'] for t in traces)
    context_hit_count = sum(t['context_hit'] for t in traces)

    summary = {
        "strategy": name,
        "split": "dev",
        "n": n,
        "hit1": hit1_count,
        "hit1_rate": round(hit1_count / n * 100, 2),
        "hit3": hit3_count,
        "hit3_rate": round(hit3_count / n * 100, 2),
        "context_hit": context_hit_count,
        "context_evidence_rate": round(context_hit_count / n * 100, 2),
        "median_ms": round(statistics.median(latencies), 2),
        "avg_ms": round(statistics.mean(latencies), 2)
    }

    return summary, traces


def main():
    print("=" * 75)
    print(" CHECKPOINT 2: RETRIEVAL ENGINES BENCHMARK & COMPARISON (DEV SPLIT)")
    print("=" * 75)

    chunks_file = DATA_DIR / "chunks.json"
    questions_file = DATA_DIR / "questions.json"

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(questions_file, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    # We evaluate on dev split (15 answerable questions)
    dev_questions = [q for q in all_questions if q.get("split") == "dev"]
    print(f"Loaded {len(chunks)} chunks, evaluating on {len(dev_questions)} dev questions.")

    # 1. Initialize engines
    print("Initializing retrieval engines...")
    tfidf_retriever = TfidfRetriever(chunks)
    bm25_retriever = BM25Retriever(chunks)
    dense_retriever = DenseRetriever(collection_name="strategy_b")
    hybrid_retriever = HybridRetriever(dense_retriever=dense_retriever, sparse_retriever=bm25_retriever, k=60)
    cohere_reranker = CohereReranker(model="rerank-v3.5")

    # Define engine runners
    def run_tfidf(q, top_k=3, filters=None):
        return tfidf_retriever.search(q, top_k=top_k, filters=filters)

    def run_bm25(q, top_k=3, filters=None):
        return bm25_retriever.search(q, top_k=top_k, filters=filters)

    def run_dense(q, top_k=3, filters=None):
        return dense_retriever.search(q, top_k=top_k, filters=filters)

    def run_hybrid(q, top_k=3, filters=None):
        return hybrid_retriever.search(q, top_k=top_k, filters=filters)

    def run_two_stage(q, top_k=3, filters=None):
        # Top-10 candidates from Hybrid search, then rerank with Cohere
        candidates = hybrid_retriever.search(q, top_k=10, filters=filters)
        return cohere_reranker.rerank(q, candidates, top_k=top_k)

    engines = [
        ("TF-IDF (Lexical Baseline)", run_tfidf),
        ("BM25 (Sparse Lexical)", run_bm25),
        ("Dense (Qdrant + OpenAI 1536d)", run_dense),
        ("Hybrid (Dense + BM25 via RRF)", run_hybrid),
        ("Two-Stage (Hybrid + Cohere Rerank)", run_two_stage)
    ]

    all_summaries = []
    all_traces = {}

    for name, runner in engines:
        summary, traces = evaluate_engine(name, runner, dev_questions, budget=800)
        all_summaries.append(summary)
        all_traces[name] = traces

        # Save individual trace
        safe_name = name.lower().replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "").replace("-", "_")
        trace_file = RESULTS_DIR / f"traces_dev_{safe_name}.json"
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(traces, f, ensure_ascii=False, indent=2)

    # Save consolidated summary
    summary_file = RESULTS_DIR / "dev_retrieval_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    # Print comparative markdown table
    print("\n" + "=" * 90)
    print(" BẢNG SO SÁNH HIỆU NĂNG RETRIEVAL TRÊN TẬP DEV (15 CÂU HỎI)")
    print("=" * 90)
    header = f"| {'Phương pháp':<36} | {'Hit@1':<12} | {'Evidence Hit@3':<16} | {'Context Evidence':<18} | {'Median ms':<10} |"
    sep = f"|{'-'*38}|{'-'*14}|{'-'*18}|{'-'*20}|{'-'*12}|"
    print(header)
    print(sep)
    for s in all_summaries:
        h1 = f"{s['hit1']}/{s['n']} ({s['hit1_rate']}%)"
        h3 = f"{s['hit3']}/{s['n']} ({s['hit3_rate']}%)"
        ctx = f"{s['context_hit']}/{s['n']} ({s['context_evidence_rate']}%)"
        lat = f"{s['median_ms']} ms"
        print(f"| {s['strategy']:<36} | {h1:<12} | {h3:<16} | {ctx:<18} | {lat:<10} |")
    print("=" * 90)
    print(f"\nTraces and summaries saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
