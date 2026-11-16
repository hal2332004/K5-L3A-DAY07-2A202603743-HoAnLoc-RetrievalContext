"""Run full Agentic RAG evaluation on a split (dev or test) using real LLM (gpt-4o-mini).
Saves traces compatible with starter/agent_metrics.py.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "starter"))

from agentic import run_agent
from src.agent import OpenAIController, RetrieverAdapter
from src.retrievers import BM25Retriever, CohereReranker, DenseRetriever, HybridRetriever, TfidfRetriever

DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(split: str):
    cases_file = DATA_DIR / "cases.json"
    benchmark_file = DATA_DIR / "benchmark_questions_all.json"

    with open(cases_file, "r", encoding="utf-8") as f:
        cases = json.load(f)
    with open(benchmark_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    split_cases = [c for c in cases if c.get("split") == split]
    q_map = {q["id"]: q for q in questions if q.get("split") == split}

    return split_cases, q_map


def get_retriever_adapter(engine_name: str, chunks: list[dict]):
    if engine_name == "tfidf":
        engine = TfidfRetriever(chunks)
    elif engine_name == "bm25":
        engine = BM25Retriever(chunks)
    elif engine_name == "dense":
        engine = DenseRetriever(collection_name="strategy_b")
    elif engine_name == "rerank":
        dense = DenseRetriever(collection_name="strategy_b")
        bm25 = BM25Retriever(chunks)
        hybrid = HybridRetriever(dense, bm25)
        reranker = CohereReranker()
        class RerankEngine:
            def search(self, query, top_k=3, filters=None):
                cands = hybrid.search(query, top_k=10, filters=filters)
                return reranker.rerank(query, cands, top_k=top_k)
        engine = RerankEngine()
    else:  # default "hybrid"
        dense = DenseRetriever(collection_name="strategy_b")
        bm25 = BM25Retriever(chunks)
        engine = HybridRetriever(dense, bm25)

    return RetrieverAdapter(engine)


def main():
    parser = argparse.ArgumentParser(description="Run agentic evaluation loop with real LLM")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--engine", choices=("hybrid", "dense", "bm25", "tfidf", "rerank"), default="hybrid")
    parser.add_argument("--budget", type=int, default=1500)
    parser.add_argument("--max_searches", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    print("=" * 75)
    print(f" AGENTIC RAG EVALUATION — SPLIT: {args.split.upper()} | ENGINE: {args.engine.upper()} | BUDGET: {args.budget}")
    print("=" * 75)

    with open(DATA_DIR / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    cases, q_map = load_dataset(args.split)
    print(f"Loaded {len(cases)} cases for split '{args.split}' ({sum(1 for c in cases if c['answerable'])} answerable, {sum(1 for c in cases if not c['answerable'])} abstain).")

    adapter = get_retriever_adapter(args.engine, chunks)
    controller = OpenAIController(model="gpt-4o-mini", temperature=0.1)

    traces = []
    latencies = []

    for idx, case in enumerate(cases, start=1):
        cid = case["id"]
        q_obj = q_map[cid]
        question = q_obj["question"]
        filters = q_obj.get("filter")
        is_answerable = case["answerable"]

        print(f"\n[{idx}/{len(cases)}] Running {cid} (Answerable: {is_answerable})...")
        print(f"Q: {question[:80]}...")

        agent_res = None
        for run_attempt in range(1, 4):
            start_time = time.perf_counter()
            try:
                agent_res = run_agent(
                    question=question,
                    controller=controller,
                    retriever=adapter,
                    filters=filters,
                    max_searches=args.max_searches,
                    top_k=args.top_k,
                    budget=args.budget
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                status = agent_res["status"]
                answer = agent_res["answer"]
                citations = agent_res["citations"]
                events = agent_res["events"]
                break
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if run_attempt < 3:
                    print(f"  [Attempt {run_attempt} failed with {e}, retrying in 2s...]")
                    time.sleep(2)
                else:
                    status = "error"
                    answer = f"Error: {e}"
                    citations = []
                    events = [{"event": "error", "error": str(e)}]

        latencies.append(elapsed_ms)
        print(f"-> Status: {status} ({elapsed_ms:.1f}ms) | Citations: {citations}")
        print(f"-> Answer excerpt: {answer[:120]}...")

        traces.append({
            "query_id": cid,
            "split": args.split,
            "question": question,
            "answerable": is_answerable,
            "status": status,
            "answer": answer,
            "citations": citations,
            "events": events,
            "elapsed_ms": round(elapsed_ms, 2)
        })

    # Save traces
    trace_file = RESULTS_DIR / f"agent_traces_{args.split}_{args.engine}.json"
    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(traces)} traces to: {trace_file}")

    # Calculate summary metrics
    total = len(traces)
    tool_use_count = sum(1 for t in traces if any(e.get("event") == "search" for e in t.get("events", [])))
    search_counts = [sum(1 for e in t.get("events", []) if e.get("event") == "search") for t in traces]
    answered_count = sum(1 for t in traces if t["status"] == "answered")
    abstain_count = sum(1 for t in traces if t["status"] == "abstain")
    error_count = sum(1 for t in traces if t["status"] == "error")

    answerable_cases = [t for t in traces if t["answerable"]]
    unanswerable_cases = [t for t in traces if not t["answerable"]]

    correct_abstention = sum(1 for t in unanswerable_cases if t["status"] == "abstain")
    false_abstention = sum(1 for t in answerable_cases if t["status"] == "abstain")

    # Citations in context check
    all_citations = []
    valid_citations = 0
    for t in traces:
        c_list = t.get("citations", [])
        all_citations.extend(c_list)
        # Check against context_ids in search events
        context_ids = set()
        for e in t.get("events", []):
            context_ids.update(e.get("context_ids", []))
        for c in c_list:
            if c in context_ids:
                valid_citations += 1

    summary = {
        "split": args.split,
        "engine": args.engine,
        "budget": args.budget,
        "total_cases": total,
        "tool_use_rate": f"{tool_use_count}/{total} ({tool_use_count/total*100:.1f}%)",
        "avg_searches_per_query": round(statistics.mean(search_counts), 2) if search_counts else 0,
        "search_distribution": {f"{s}_searches": search_counts.count(s) for s in sorted(set(search_counts))},
        "answered": f"{answered_count}/{total} ({answered_count/total*100:.1f}%)",
        "abstain_total": f"{abstain_count}/{total} ({abstain_count/total*100:.1f}%)",
        "correct_abstention_on_unanswerable": f"{correct_abstention}/{len(unanswerable_cases)} ({correct_abstention/len(unanswerable_cases)*100:.1f}%)" if unanswerable_cases else "N/A",
        "false_abstention_on_answerable": f"{false_abstention}/{len(answerable_cases)} ({false_abstention/len(answerable_cases)*100:.1f}%)" if answerable_cases else "N/A",
        "citation_context_validity": f"{valid_citations}/{len(all_citations)} ({valid_citations/len(all_citations)*100:.1f}%)" if all_citations else "0/0",
        "answer_citation_coverage": f"{sum(1 for t in traces if t['status'] == 'answered' and t['citations'])}/{answered_count}",
        "provider_errors": error_count,
        "median_ms": round(statistics.median(latencies), 2),
        "total_prompt_tokens": controller.total_prompt_tokens,
        "total_completion_tokens": controller.total_completion_tokens,
        "est_cost_usd": round((controller.total_prompt_tokens * 0.15 + controller.total_completion_tokens * 0.60) / 1_000_000, 5)
    }

    summary_file = RESULTS_DIR / f"agent_summary_{args.split}_{args.engine}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(" AGENTIC EVALUATION SUMMARY REPORT")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:<35}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
