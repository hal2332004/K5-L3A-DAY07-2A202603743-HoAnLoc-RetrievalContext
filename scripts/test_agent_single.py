import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "starter"))

from agentic import run_agent
from src.agent import OpenAIController, RetrieverAdapter
from src.retrievers import BM25Retriever, DenseRetriever, HybridRetriever

with open("data/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

dense = DenseRetriever()
bm25 = BM25Retriever(chunks)
hybrid = HybridRetriever(dense, bm25)
adapter = RetrieverAdapter(hybrid)
controller = OpenAIController()

print("--- Testing Answerable Query: q01 ---")
q1 = "What are the three layers of any AI application stack according to Chip Huyen?"
res1 = run_agent(q1, controller, adapter, max_searches=2, top_k=3, budget=1500)
print(f"Status: {res1['status']}")
print(f"Answer: {res1['answer']}")
print(f"Citations: {res1['citations']}")
print(f"Events: {len(res1['events'])}")

print("\n--- Testing Unanswerable Query: q_abs_dev_01 ---")
q2 = "How does Megatron-LM implement 3D tensor parallelism across 8x H100 GPUs in Chapter 3?"
res2 = run_agent(q2, controller, adapter, max_searches=2, top_k=3, budget=1500)
print(f"Status: {res2['status']}")
print(f"Answer: {res2['answer']}")
print(f"Citations: {res2['citations']}")
print(f"Events: {len(res2['events'])}")
