import json
import statistics
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.retrievers import BM25Retriever, DenseRetriever, HybridRetriever

with open(ROOT_DIR / "data" / "chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

with open(ROOT_DIR / "data" / "questions.json", encoding="utf-8") as f:
    questions = [q for q in json.load(f) if q.get("split") == "test"]

def hit(chunk, gold):
    return any(
        chunk.get("source_id") == g["source_id"]
        and chunk.get("page") == g["page"]
        and g["evidence"] in chunk.get("text", "")
        for g in gold
    )

def pack_context(results, budget=1500):
    blocks, ids, seen = [], [], set()
    for c in results:
        cid = c.get("id")
        if cid in seen:
            continue
        seen.add(cid)
        block = f"[{cid} | {c.get('source')} | page {c.get('page')}]\n{c.get('text')}"
        cand = "\n\n".join(blocks + [block])
        if len(cand) <= budget:
            blocks.append(block)
            ids.append(cid)
    return "\n\n".join(blocks), ids

dense = DenseRetriever(collection_name="strategy_b")
bm25 = BM25Retriever(chunks)
hybrid = HybridRetriever(dense, bm25)

latencies = []
hit1_count = 0
hit3_count = 0
ctx_hit_1500 = 0
ctx_hit_800 = 0

for q in questions:
    gold = q["gold"]
    t0 = time.perf_counter()
    results = hybrid.search(q["question"], top_k=3, filters=q.get("filter"))
    elapsed = (time.perf_counter() - t0) * 1000
    latencies.append(elapsed)

    if results and hit(results[0], gold):
        hit1_count += 1
    if any(hit(r, gold) for r in results[:3]):
        hit3_count += 1

    _, ids_1500 = pack_context(results, budget=1500)
    if any(hit(r, gold) for r in results if r["id"] in ids_1500):
        ctx_hit_1500 += 1

    _, ids_800 = pack_context(results, budget=800)
    if any(hit(r, gold) for r in results if r["id"] in ids_800):
        ctx_hit_800 += 1

n = len(questions)
res = {
    "split": "test",
    "strategy": "hybrid_dense_plus_bm25",
    "n": n,
    "hit1": f"{hit1_count}/{n} ({hit1_count/n*100:.1f}%)",
    "hit3": f"{hit3_count}/{n} ({hit3_count/n*100:.1f}%)",
    "ctx_evidence_rate_budget_1500": f"{ctx_hit_1500}/{n} ({ctx_hit_1500/n*100:.1f}%)",
    "ctx_evidence_rate_budget_800": f"{ctx_hit_800}/{n} ({ctx_hit_800/n*100:.1f}%)",
    "median_ms": round(statistics.median(latencies), 2),
    "cost_usd": round(n * 0.000003, 6)
}

print(json.dumps(res, indent=2))
with open(ROOT_DIR / "results" / "selected_test_retrieval.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
