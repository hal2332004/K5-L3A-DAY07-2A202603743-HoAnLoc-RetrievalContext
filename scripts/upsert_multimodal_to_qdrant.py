"""Upsert 15 Multimodal Visual/Table Chunks into Qdrant Cloud collection 'strategy_b'.

This ensures the Dense Retriever and Qdrant Cloud Vector Store contain all 364 chunks
(349 base text chunks + 15 visual table/diagram/chart chunks).
"""
import os
import json
import uuid
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not qdrant_url or not qdrant_api_key or not openai_api_key:
    raise ValueError("Missing environment variables in .env (QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY)")

qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, check_compatibility=False, timeout=30)
openai_client = OpenAI(api_key=openai_api_key)

COLLECTION_NAME = "strategy_b"

# 1. Load Chunks
multimodal_path = ROOT_DIR / "data" / "chunks_multimodal_advanced.json"
with open(multimodal_path, "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

# Extract only the 15 multimodal assets (indices 349 to 363)
multimodal_chunks = all_chunks[349:]
print(f"Found {len(multimodal_chunks)} multimodal chunks to embed and upsert into Qdrant '{COLLECTION_NAME}'...")

# 2. Check current collection count
info_before = qdrant_client.get_collection(COLLECTION_NAME)
print(f"Current points in '{COLLECTION_NAME}': {info_before.points_count}")

# 3. Generate Embeddings for the 15 chunks using text-embedding-3-small
texts_to_embed = [c["text"].replace("\n", " ") for c in multimodal_chunks]
print("Requesting embeddings from OpenAI text-embedding-3-small...")

t0 = time.perf_counter()
resp = openai_client.embeddings.create(
    input=texts_to_embed,
    model="text-embedding-3-small"
)
vectors = [d.embedding for d in resp.data]
elapsed_embed = time.perf_counter() - t0
print(f"Generated {len(vectors)} embeddings (1536d) in {elapsed_embed:.2f}s.")

# 4. Prepare Qdrant Points
points = []
for c, vec in zip(multimodal_chunks, vectors):
    pt_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, c["id"]))
    meta = c.get("metadata", {})
    payload = {
        "chunk_id": c["id"],
        "id": c["id"],
        "document_id": c.get("source_id", "chip-huyen-ai-engineering-ch1-2"),
        "source": c.get("source", "Chip Huyen - AI Engineering"),
        "page": c.get("page", 0),
        "text": c.get("text", ""),
        "chapter": meta.get("chapter", 1),
        "section": meta.get("section", "multimodal_asset"),
        "type": meta.get("type", "visual_asset"),
        "asset_name": meta.get("asset_name", ""),
        "char_count": len(c.get("text", "")),
        "char_length": len(c.get("text", "")),
        "chunk_strategy": "multimodal_visual_asset"
    }
    points.append(models.PointStruct(id=pt_id, vector=vec, payload=payload))

# 5. Upsert into Qdrant Cloud
print(f"Upserting {len(points)} points into '{COLLECTION_NAME}'...")
qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

# 6. Verify collection count
info_after = qdrant_client.get_collection(COLLECTION_NAME)
print(f"Updated points in '{COLLECTION_NAME}': {info_after.points_count} (added {info_after.points_count - info_before.points_count})")

# 7. Test semantic vector search on one multimodal asset
print("\nTesting semantic search for: 'What is the feedforward dimension of Llama 3-70B in Table 2-4?'...")
q_resp = openai_client.embeddings.create(
    input=["What is the feedforward dimension of Llama 3-70B in Table 2-4?"],
    model="text-embedding-3-small"
)
q_vec = q_resp.data[0].embedding

search_res = qdrant_client.query_points(
    collection_name=COLLECTION_NAME,
    query=q_vec,
    limit=3,
    with_payload=True
)

print(f"Top 3 Qdrant search results:")
for rank, pt in enumerate(search_res.points, 1):
    cid = pt.payload.get("chunk_id") or pt.payload.get("id")
    page = pt.payload.get("page")
    asset = pt.payload.get("asset_name")
    print(f"  #{rank}: {cid} (Page {page}, Asset: {asset}, Score: {pt.score:.4f})")
    print(f"     Text snippet: {pt.payload.get('text', '')[:120]}...")
