import json
import re
from extract_day6_data import extract_chunks

chunks = extract_chunks()
with open("D:/Ky 4/VinAI/K5-L3B-DAY06-2A202603743-DataIngestion/benchmark/questions.json", encoding="utf-8") as f:
    qs = json.load(f)

for q in qs:
    eq = q["evidence_quote"]
    p = q["gold_page"]
    p_chunks = [c for c in chunks if c["page"] == p]
    verbatim = any(eq in c["text"] for c in p_chunks)
    ws_norm = any(re.sub(r"\s+", " ", eq) in re.sub(r"\s+", " ", c["text"]) for c in p_chunks)
    print(f"{q['id']} (p.{p}): verbatim={verbatim}, ws_norm={ws_norm}")
    if not verbatim and ws_norm:
        # Show how it appears in chunk text
        clean_eq = re.sub(r"\s+", " ", eq)
        for c in p_chunks:
            c_norm = re.sub(r"\s+", " ", c["text"])
            if clean_eq in c_norm:
                # Find the actual substring in c["text"]
                words = eq.split()
                first, last = words[0], words[-1]
                pattern = re.escape(first) + r".*?" + re.escape(last)
                m = re.search(pattern, c["text"], re.DOTALL)
                if m:
                    print(f"   Chunk actual: {repr(m.group(0))}")
                break
