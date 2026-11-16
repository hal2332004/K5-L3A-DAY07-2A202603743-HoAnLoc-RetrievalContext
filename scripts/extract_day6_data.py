"""Extract Day 6 chunks and benchmark questions, generate abstain cases, and produce Day 7 JSON datasets.
"""
import json
import re
import unicodedata
from pathlib import Path
import pypdf

DAY6_ROOT = Path("D:/Ky 4/VinAI/K5-L3B-DAY06-2A202603743-DataIngestion")
DAY7_ROOT = Path(__file__).resolve().parents[1]

PDF_PATH = DAY6_ROOT / "data/Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf"
DAY6_BENCHMARK = DAY6_ROOT / "benchmark/questions.json"

DOCUMENT_ID = "chip-huyen-ai-engineering-ch1-2"
SOURCE_NAME = "Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf"


def get_section_for_page(page: int) -> tuple[int, str]:
    if page <= 72:
        chapter = 1
        if page < 40:
            section = "the_rise_of_ai_engineering"
        elif page < 52:
            section = "foundation_model_use_cases"
        elif page < 61:
            section = "planning_ai_applications"
        else:
            section = "ai_engineering_stack"
    else:
        chapter = 2
        if page < 82:
            section = "training_data"
        elif page < 102:
            section = "model_architecture"
        elif page < 112:
            section = "post_training"
        elif page < 120:
            section = "sampling_strategies"
        elif page < 123:
            section = "test_time_compute"
        elif page < 129:
            section = "structured_outputs"
        else:
            section = "probabilistic_nature_of_ai"
    return chapter, section


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\r\n", "\n")
    text = re.sub(r'(\w+)[\u2010\u2013\u2014-]\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def by_paragraph(page_text: str, max_size: int = 800) -> list[str]:
    raw_paras = [p.strip() for p in re.split(r'\n\s*\n', page_text) if p.strip()]
    if not raw_paras:
        raw_paras = [page_text.strip()]

    processed_paras: list[str] = []
    for p in raw_paras:
        if len(p) <= max_size:
            processed_paras.append(p)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', p)
            current_sub = ""
            for s in sentences:
                if not current_sub:
                    current_sub = s
                elif len(current_sub) + 1 + len(s) <= max_size:
                    current_sub += " " + s
                else:
                    processed_paras.append(current_sub.strip())
                    current_sub = s
            if current_sub.strip():
                processed_paras.append(current_sub.strip())

    chunks: list[str] = []
    current_chunk = ""
    for para in processed_paras:
        if not current_chunk:
            current_chunk = para
        elif len(current_chunk) + 2 + len(para) <= max_size:
            current_chunk += "\n\n" + para
        else:
            chunks.append(current_chunk)
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def extract_chunks() -> list[dict]:
    print(f"Reading PDF from: {PDF_PATH}")
    reader = pypdf.PdfReader(str(PDF_PATH))
    chunks = []
    start_page = 25
    end_page = 136

    for p_num in range(start_page, end_page + 1):
        raw_text = reader.pages[p_num - 1].extract_text() or ""
        cleaned = normalize_text(raw_text)
        if not cleaned:
            continue
        chapter, section = get_section_for_page(p_num)
        para_chunks = by_paragraph(cleaned, max_size=800)
        for idx, text in enumerate(para_chunks):
            chunk_id = f"{DOCUMENT_ID}_p{p_num}_p{idx:03d}"
            chunks.append({
                "id": chunk_id,
                "source_id": DOCUMENT_ID,
                "source": SOURCE_NAME,
                "page": p_num,
                "text": text,
                "metadata": {
                    "chapter": chapter,
                    "section": section,
                    "char_length": len(text),
                    "chunk_index": idx
                }
            })
    print(f"Extracted {len(chunks)} chunks across physical pages {start_page}..{end_page}.")
    return chunks


def find_exact_chunk_substring(quote: str, page: int, chunks: list[dict]) -> str:
    p_chunks = [c for c in chunks if c["page"] == page]
    # 1. Exact match
    for c in p_chunks:
        if quote in c["text"]:
            return quote

    # 2. Regex word sequence matching
    clean_quote = re.sub(r"\s+", " ", quote).strip()
    words = clean_quote.split()
    pattern_str = r"\s+".join(re.escape(w) for w in words)
    pattern = re.compile(pattern_str)
    for c in p_chunks:
        m = pattern.search(c["text"])
        if m:
            return m.group(0)

    # 3. Relax apostrophes/quotes
    relaxed_words = []
    for w in words:
        w_esc = re.escape(w).replace(r"\'", r"['’]").replace("’", r"['’]")
        relaxed_words.append(w_esc)
    relaxed_pattern = re.compile(r"\s+".join(relaxed_words))
    for c in p_chunks:
        m = relaxed_pattern.search(c["text"])
        if m:
            return m.group(0)

    raise ValueError(f"Could not locate verbatim quote for page {page}: {quote}")


def build_questions_and_cases(chunks: list[dict]):
    with open(DAY6_BENCHMARK, "r", encoding="utf-8") as f:
        day6_qs = json.load(f)

    # 1. 20 Answerable questions (15 dev + 5 test)
    questions = []
    for q in day6_qs:
        exact_evidence = find_exact_chunk_substring(q["evidence_quote"], q["gold_page"], chunks)
        converted = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "gold": [{
                "source_id": q.get("gold_source_id", DOCUMENT_ID),
                "page": q["gold_page"],
                "evidence": exact_evidence
            }],
            "expected_answer": q["expected_answer"]
        }
        if q.get("filter"):
            converted["filter"] = q["filter"]
        questions.append(converted)

    # Double check all 20
    for q in questions:
        g = q["gold"][0]
        found = any(c["source_id"] == g["source_id"] and c["page"] == g["page"] and g["evidence"] in c["text"] for c in chunks)
        if not found:
            raise ValueError(f"Validation failed for {q['id']}")

    print(f"Verified all {len(questions)} answerable questions against chunk texts: 100% matched!")

    # 2. 4 Unanswerable (Abstain) questions (2 dev + 2 test)
    abstain_questions = [
        {
            "id": "q_abs_dev_01",
            "split": "dev",
            "question": "How does Megatron-LM implement 3D tensor parallelism across 8x H100 GPUs in Chapter 3?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "The corpus only covers Chapters 1 & 2 (pages 25-136); Chapter 3 on distributed training is not in the ingested textbook."
        },
        {
            "id": "q_abs_dev_02",
            "split": "dev",
            "question": "What is O'Reilly Media's student refund policy for damaged physical copies of this book?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Commercial publisher refund and exchange policies are not part of technical engineering book contents."
        },
        {
            "id": "q_abs_test_01",
            "split": "test",
            "question": "What is the exact volume discount percentage offered by OpenAI for non-profit organizations deploying GPT-4o?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Specific enterprise or non-profit API pricing discount rates are external vendor commercial policies not mentioned in the book."
        },
        {
            "id": "q_abs_test_02",
            "split": "test",
            "question": "According to Chapter 2, what is the exact hardware spec of Google TPU v5e pod optical interconnect bandwidth?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Chapter 2 introduces foundation models conceptually; TPU v5e pod interconnect bandwidth hardware specifications are absent."
        }
    ]

    # 3. Build cases.json for all 24 questions (for agent_metrics.py)
    cases = []
    for q in questions:
        cases.append({
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer"
        })
    for q in abstain_questions:
        cases.append({
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": False,
            "expected_behavior": q["expected_behavior"],
            "absence_reason": q["absence_reason"]
        })

    return questions, abstain_questions, cases


def main():
    print("=" * 65)
    print(" CHECKPOINT 1: DATA INGESTION & BENCHMARK PREPARATION")
    print("=" * 65)

    data_dir = DAY7_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    chunks = extract_chunks()
    questions, abstain_questions, cases = build_questions_and_cases(chunks)

    # Save chunks.json
    chunks_path = data_dir / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to: {chunks_path}")

    # Save questions.json (The 20 gold answerable questions for baseline retrieval)
    questions_path = data_dir / "questions.json"
    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(questions)} answerable questions to: {questions_path}")

    # Save all 24 cases (17 dev + 7 test) to cases.json
    cases_path = data_dir / "cases.json"
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(cases)} total cases (20 answerable + 4 abstain) to: {cases_path}")

    # Also save full 24 benchmark questions (with abstain questions included) to benchmark_questions_all.json
    all_questions_path = data_dir / "benchmark_questions_all.json"
    all_questions = list(questions) + [
        {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": False,
            "expected_answer": "Không đủ bằng chứng trong tài liệu để trả lời.",
            "gold": [],
            "absence_reason": q["absence_reason"]
        } for q in abstain_questions
    ]
    with open(all_questions_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    print(f"Saved all 24 questions (with abstain) to: {all_questions_path}")

    print("\nDataset summary:")
    print(f"- Total chunks: {len(chunks)}")
    print(f"- Dev cases: {sum(1 for c in cases if c['split'] == 'dev')} (15 answerable + 2 abstain)")
    print(f"- Test cases: {sum(1 for c in cases if c['split'] == 'test')} (5 answerable + 2 abstain)")
    print(f"- Total cases: {len(cases)}")
    print("Checkpoint 1 Data Generation Completed Successfully!")


if __name__ == "__main__":
    main()
