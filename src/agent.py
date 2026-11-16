"""Agentic Controller and Retriever Adapter for Day 7.
Integrates real LLM (gpt-4o-mini) with tool decision loop and citation enforcement.
"""
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


class OpenAIController:
    """Real LLM Controller using OpenAI gpt-4o-mini with structured JSON actions."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None, temperature: float = 0.1):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment or .env")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Decide the next action based on question and accumulated context in history."""
        system_prompt = (
            "You are an expert AI Engineering research assistant with access to a textbook "
            "'AI Engineering: Building Applications with Foundation Models' by Chip Huyen.\n\n"
            "You have access to a retrieval search tool to find facts.\n"
            "Analyze the user question and the current retrieval context history.\n\n"
            "You MUST respond with a JSON object in one of two formats:\n"
            "1. If context is empty OR insufficient and a better query could find the answer:\n"
            '   {"action": "search", "query": "<focused search query string>"}\n'
            "2. If context contains sufficient evidence to answer:\n"
            '   {"action": "answer", "answer": "<concise answer text citing [chunk-id] directly next to facts>", "citations": ["chunk-id-1", "chunk-id-2"]}\n'
            "3. If the context does not contain the answer, and the topic is clearly outside the textbook:\n"
            '   {"action": "answer", "answer": "Không đủ bằng chứng trong tài liệu để trả lời.", "citations": []}\n\n'
            "CRITICAL RULES:\n"
            "- Only answer using facts from the provided context excerpts.\n"
            "- Every citation ID in 'citations' MUST match an ID in the context.\n"
            "- Every cited ID MUST also appear verbatim as [chunk-id] in the answer text.\n"
            "- If no evidence exists in the context, output empty citations: [] and state that information is insufficient."
        )

        messages = [{"role": "system", "content": system_prompt}]

        if not history:
            user_msg = (
                f"User Question: {question}\n\n"
                "Current context is empty. Please generate a search action to retrieve relevant textbook excerpts."
            )
        else:
            latest = history[-1]
            queries_run = latest.get("queries", [])
            context = latest.get("context", "")
            context_ids = latest.get("context_ids", [])

            user_msg = (
                f"User Question: {question}\n\n"
                f"Previous queries executed: {queries_run}\n"
                f"Available chunk IDs in context: {context_ids}\n\n"
                f"Cumulative Retrieved Context:\n{context}\n\n"
                "Based on this context, determine if you have enough evidence to answer, need another search, "
                "or if there is insufficient evidence."
            )

        messages.append({"role": "user", "content": user_msg})

        response = None
        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                    timeout=30.0
                )
                break
            except Exception as e:
                if attempt < 3:
                    import time
                    time.sleep(2 * attempt)
                else:
                    raise e

        if response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens

        content = response.choices[0].message.content or "{}"
        try:
            action_data = json.loads(content)
        except Exception:
            action_data = {"action": "search", "query": question}

        # Normalize and validate action format
        action_type = action_data.get("action")
        if action_type == "search":
            query = action_data.get("query", "").strip()
            if not query:
                query = question
            return {"action": "search", "query": query}
        elif action_type == "answer":
            answer = action_data.get("answer", "").strip()
            citations = action_data.get("citations", [])
            if not isinstance(citations, list):
                citations = []
            citations = [str(c).strip() for c in citations if str(c).strip()]

            # Ensure citations that appear in answer are preserved
            if not citations and answer:
                found_in_text = re.findall(r'\[([^\]]+)\]', answer)
                if history:
                    valid_ids = set(history[-1].get("context_ids", []))
                    citations = [cid for cid in found_in_text if cid in valid_ids]

            return {"action": "answer", "answer": answer, "citations": citations}
        else:
            # Fallback to answer or search
            if history:
                return {
                    "action": "answer",
                    "answer": "Không đủ bằng chứng trong tài liệu để trả lời.",
                    "citations": []
                }
            return {"action": "search", "query": question}


class RetrieverAdapter:
    """Adapts Day 7 retrievers (BM25, Dense, Hybrid, Rerank) to the Retriever protocol expected by agentic.py."""

    def __init__(self, engine: Any):
        self.engine = engine
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "filters": filters, "top_k": top_k})
        results = self.engine.search(query, top_k=top_k, filters=filters)

        # Deduplicate and ensure top_k limit
        unique_results = []
        seen_ids = set()
        for r in results:
            cid = r.get("id", r.get("chunk_id"))
            if cid not in seen_ids:
                seen_ids.add(cid)
                item = dict(r)
                item["id"] = cid
                unique_results.append(item)
            if len(unique_results) >= top_k:
                break
        return unique_results
