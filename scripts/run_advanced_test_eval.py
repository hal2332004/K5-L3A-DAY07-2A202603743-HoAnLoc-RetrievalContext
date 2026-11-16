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

print("Loading Advanced Benchmark Test Split and Multimodal Chunks...", flush=True)

with open(ROOT_DIR / "data" / "advanced_cases.json", encoding="utf-8") as f:
    cases = [c for c in json.load(f) if c.get("split") == "test"]

with open(ROOT_DIR / "data" / "advanced_benchmark.json", encoding="utf-8") as f:
    q_map = {q["id"]: q for q in json.load(f) if q.get("split") == "test"}

with open(ROOT_DIR / "data" / "chunks_multimodal_advanced.json", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(cases)} test cases and {len(chunks)} multimodal chunks.", flush=True)

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
    group = case["main_group"]
    is_answerable = case["answerable"]

    print(f"\n[{idx}/{len(cases)}] [{group}] {cid} (Answerable: {is_answerable})...", flush=True)
    print(f"Q: {question}", flush=True)

    agent_res = None
    for attempt in range(1, 4):
        t0 = time.perf_counter()
        try:
            agent_res = run_agent(
                question=question,
                controller=controller,
                retriever=adapter,
                filters=None,
                max_searches=3,
                top_k=3,
                budget=1500
            )
            elapsed = (time.perf_counter() - t0) * 1000
            status = agent_res["status"]
            answer = agent_res["answer"]
            citations = agent_res["citations"]
            events = agent_res["events"]
            break
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            if attempt < 3:
                print(f"  [Attempt {attempt} failed with {e}, retrying in 2s...]", flush=True)
                time.sleep(2)
            else:
                status = "error"
                answer = f"Error: {e}"
                citations = []
                events = [{"event": "error", "error": str(e)}]

    print(f"-> Status: {status} ({elapsed:.1f}ms) | Citations: {citations}", flush=True)
    print(f"-> Answer excerpt: {answer[:120]}...", flush=True)

    traces.append({
        "query_id": cid,
        "split": "test",
        "main_group": group,
        "question": question,
        "answerable": is_answerable,
        "status": status,
        "answer": answer,
        "citations": citations,
        "events": events,
        "elapsed_ms": round(elapsed, 2)
    })

out_file = ROOT_DIR / "results" / "advanced_test_traces.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(traces, f, ensure_ascii=False, indent=2)

print(f"\nSuccessfully wrote {len(traces)} traces to {out_file}", flush=True)
