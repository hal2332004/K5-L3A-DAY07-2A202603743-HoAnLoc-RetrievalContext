import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "starter"))

from agentic import run_agent
from src.agent import OpenAIController, RetrieverAdapter
from src.retrievers import BM25Retriever, DenseRetriever, HybridRetriever

print("Loading test cases and chunks...", flush=True)
with open(ROOT_DIR / "data" / "cases.json", encoding="utf-8") as f:
    cases = [c for c in json.load(f) if c.get("split") == "test"]

with open(ROOT_DIR / "data" / "benchmark_questions_all.json", encoding="utf-8") as f:
    q_map = {q["id"]: q for q in json.load(f) if q.get("split") == "test"}

with open(ROOT_DIR / "data" / "chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(cases)} test cases. Initializing Hybrid retriever...", flush=True)
dense = DenseRetriever(collection_name="strategy_b")
bm25 = BM25Retriever(chunks)
hybrid = HybridRetriever(dense, bm25)
adapter = RetrieverAdapter(hybrid)
controller = OpenAIController(model="gpt-4o-mini", temperature=0.1)

traces = []
for idx, case in enumerate(cases, 1):
    cid = case["id"]
    q_obj = q_map[cid]
    question = q_obj["question"]
    filters = q_obj.get("filter")
    is_answerable = case["answerable"]

    print(f"\n[{idx}/{len(cases)}] Query {cid} (Answerable: {is_answerable})...", flush=True)
    print(f"Q: {question[:80]}...", flush=True)

    t0 = time.perf_counter()
    try:
        res = run_agent(
            question=question,
            controller=controller,
            retriever=adapter,
            filters=filters,
            max_searches=3,
            top_k=3,
            budget=1500
        )
        elapsed = (time.perf_counter() - t0) * 1000
        status = res["status"]
        answer = res["answer"]
        citations = res["citations"]
        events = res["events"]
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        status = "error"
        answer = f"Error: {e}"
        citations = []
        events = [{"event": "error", "error": str(e)}]

    print(f"-> Status: {status} ({elapsed:.1f}ms) | Citations: {citations}", flush=True)
    print(f"-> Answer: {answer[:100]}...", flush=True)

    traces.append({
        "query_id": cid,
        "split": "test",
        "question": question,
        "answerable": is_answerable,
        "status": status,
        "answer": answer,
        "citations": citations,
        "events": events,
        "elapsed_ms": round(elapsed, 2)
    })

out_file = ROOT_DIR / "results" / "agent_traces_test_hybrid.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(traces, f, ensure_ascii=False, indent=2)

print(f"\nSuccessfully wrote {len(traces)} traces to {out_file}", flush=True)

summary = {
    "total": len(traces),
    "answered": sum(1 for t in traces if t["status"] == "answered"),
    "abstain": sum(1 for t in traces if t["status"] == "abstain"),
    "error": sum(1 for t in traces if t["status"] == "error"),
    "citations_count": sum(len(t["citations"]) for t in traces)
}
print("Summary:", summary, flush=True)
