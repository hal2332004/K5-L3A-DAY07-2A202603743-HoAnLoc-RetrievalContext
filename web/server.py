"""Day 7 Agentic & Hybrid RAG Web Server.

Endpoints:
- GET  /                     : Serves web/index.html
- GET  /api/benchmark        : Returns all 24 benchmark questions (17 dev + 7 test)
- GET  /api/summary          : Returns retrieval and agent evaluation metrics
- POST /api/chat             : Handles user question via selected engine (sparse, dense, hybrid, rerank, agentic)
"""
import http.server
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "starter"))

from agentic import run_agent
from src.agent import OpenAIController, RetrieverAdapter
from src.retrievers import BM25Retriever, CohereReranker, DenseRetriever, HybridRetriever, TfidfRetriever

# Load environment
load_dotenv(project_root / ".env")

# 1. Load Chunks (prefer multimodal advanced chunks if available)
multimodal_chunks_path = project_root / "data/chunks_multimodal_advanced.json"
chunks_path = project_root / "data/chunks.json"

if multimodal_chunks_path.exists():
    with open(multimodal_chunks_path, "r", encoding="utf-8") as f:
        CHUNKS = json.load(f)
    print(f"[Server] Loaded {len(CHUNKS)} multimodal chunks.")
elif chunks_path.exists():
    with open(chunks_path, "r", encoding="utf-8") as f:
        CHUNKS = json.load(f)
    print(f"[Server] Loaded {len(CHUNKS)} base chunks.")
else:
    raise FileNotFoundError("data/chunks.json not found! Run scripts/extract_day6_data.py first.")

# 2. Initialize Engines
print("[Server] Initializing Retrieval Engines...")
bm25_engine = BM25Retriever(CHUNKS)
tfidf_engine = TfidfRetriever(CHUNKS)
dense_engine = DenseRetriever(collection_name="strategy_b")
hybrid_engine = HybridRetriever(dense_retriever=dense_engine, sparse_retriever=bm25_engine, k=60)
cohere_reranker = CohereReranker(model="rerank-v3.5")

# 3. Initialize Agent Controller
agent_controller = OpenAIController(model="gpt-4o-mini", temperature=0.1)
agent_retriever_adapter = RetrieverAdapter(hybrid_engine)

# 4. Load Benchmark Questions
bench_path = project_root / "data/benchmark_questions_all.json"
BENCHMARK_QUESTIONS = []
if bench_path.exists():
    with open(bench_path, "r", encoding="utf-8") as f:
        BENCHMARK_QUESTIONS = json.load(f)


class Day7ServerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html_path = Path(__file__).parent / "index.html"
            if html_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "index.html not found")
        elif self.path in ("/bonus_report", "/bonus_report.html"):
            report_path = Path(__file__).parent / "bonus_report.html"
            if report_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(report_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "bonus_report.html not found")
        elif self.path == "/api/benchmark":
            self.send_json(BENCHMARK_QUESTIONS)
        elif self.path == "/api/benchmark/advanced":
            adv_bench_file = project_root / "data/advanced_benchmark.json"
            if adv_bench_file.exists():
                with open(adv_bench_file, "r", encoding="utf-8") as f:
                    self.send_json(json.load(f))
            else:
                self.send_json([])
        elif self.path == "/api/audit/advanced":
            audit_file = project_root / "results/advanced_test_audit.json"
            if audit_file.exists():
                with open(audit_file, "r", encoding="utf-8") as f:
                    self.send_json(json.load(f))
            else:
                self.send_json({})
        elif self.path == "/api/summary":
            # Consolidate results for benchmark dashboard
            retrieval_file = project_root / "results/dev_retrieval_summary.json"
            agent_file = project_root / "results/agent_summary_dev_hybrid.json"
            summary_payload = {}
            if retrieval_file.exists():
                with open(retrieval_file, "r", encoding="utf-8") as f:
                    summary_payload["retrieval_comparison"] = json.load(f)
            if agent_file.exists():
                with open(agent_file, "r", encoding="utf-8") as f:
                    summary_payload["agent_summary"] = json.load(f)
            self.send_json(summary_payload)
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                query = data.get("query", "").strip()
                mode = data.get("mode", "agentic").lower()
                top_k = int(data.get("top_k", 3))
                budget = int(data.get("budget", 1500))
                custom_filter = data.get("filter")

                if not query:
                    self.send_json({"error": "Empty query"}, status=400)
                    return

                t0 = time.perf_counter()

                # MODE 1: AGENTIC RAG (Autonomous Loop with Tool Journey)
                if mode == "agentic":
                    # Run agentic loop
                    agent_res = run_agent(
                        question=query,
                        controller=agent_controller,
                        retriever=agent_retriever_adapter,
                        filters=custom_filter,
                        max_searches=3,
                        top_k=top_k,
                        budget=budget
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                    # Extract all hits accumulated across search events
                    all_hits = []
                    seen = set()
                    for ev in agent_res.get("events", []):
                        if ev.get("event") == "search":
                            for r in ev.get("results", []):
                                cid = r.get("id")
                                if cid not in seen:
                                    seen.add(cid)
                                    all_hits.append(r)

                    payload = {
                        "query": query,
                        "mode": "agentic",
                        "status": agent_res["status"],
                        "answer": agent_res["answer"],
                        "citations": agent_res["citations"],
                        "events": agent_res["events"],
                        "hits": all_hits[:top_k * 2],
                        "elapsed_ms": round(elapsed_ms, 2)
                    }
                    self.send_json(payload)
                    return

                # NON-AGENTIC DIRECT RETRIEVAL MODES
                if mode == "sparse":
                    hits = bm25_engine.search(query, top_k=top_k, filters=custom_filter)
                elif mode == "dense":
                    hits = dense_engine.search(query, top_k=top_k, filters=custom_filter)
                elif mode == "rerank":
                    cands = hybrid_engine.search(query, top_k=10, filters=custom_filter)
                    hits = cohere_reranker.rerank(query, cands, top_k=top_k)
                else:  # "hybrid"
                    hits = hybrid_engine.search(query, top_k=top_k, filters=custom_filter)

                # Generate Answer using direct LLM call with context
                context_blocks = []
                context_ids = []
                for idx, h in enumerate(hits, start=1):
                    cid = h.get("id", f"chunk_{idx}")
                    p = h.get("page", "?")
                    txt = h.get("text", "")
                    context_blocks.append(f"[{cid} | page {p}]\n{txt}")
                    context_ids.append(cid)

                context_str = "\n\n".join(context_blocks)
                sys_prompt = (
                    "You are an expert AI Engineering assistant with authoritative knowledge from Chip Huyen's "
                    "'AI Engineering: Building Applications with Foundation Models' (Chapters 1 & 2).\n"
                    "Use ONLY the following context to answer accurately. Cite [chunk-id] inline next to facts.\n"
                    "If the answer cannot be found in the context, respond: 'Không đủ bằng chứng trong tài liệu để trả lời.'"
                )
                user_msg = f"Retrieved Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

                resp = agent_controller.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.1
                )
                answer_text = resp.choices[0].message.content or ""
                elapsed_ms = (time.perf_counter() - t0) * 1000

                # Detect status
                is_abstain = "không đủ bằng chứng" in answer_text.lower() or not hits

                payload = {
                    "query": query,
                    "mode": mode,
                    "status": "abstain" if is_abstain else "answered",
                    "answer": answer_text,
                    "citations": context_ids if not is_abstain else [],
                    "events": [
                        {
                            "event": "search",
                            "query": query,
                            "filter": custom_filter,
                            "results": hits,
                            "context": context_str,
                            "context_ids": context_ids
                        }
                    ],
                    "hits": hits,
                    "elapsed_ms": round(elapsed_ms, 2)
                }
                self.send_json(payload)

            except Exception as e:
                self.send_json({"error": str(e)}, status=500)
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def send_json(self, payload: Any, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def main():
    port = int(os.getenv("PORT", 8000))
    server = http.server.HTTPServer(("0.0.0.0", port), Day7ServerHandler)
    print("=" * 70)
    print(f" DAY 7 AGENTIC & HYBRID RAG TESTBENCH: http://localhost:{port}")
    print(" Benchmark questions, Tool Traces, and Multi-strategy Live!")
    print("=" * 70)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()
