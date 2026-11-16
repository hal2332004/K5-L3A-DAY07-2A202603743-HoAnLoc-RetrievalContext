import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

traces_path = ROOT_DIR / "results" / "advanced_test_traces.json"
bench_path = ROOT_DIR / "data" / "advanced_benchmark.json"

if not traces_path.exists():
    print(f"Traces file not found yet: {traces_path}")
    exit(0)

with open(traces_path, "r", encoding="utf-8") as f:
    traces = json.load(f)

with open(bench_path, "r", encoding="utf-8") as f:
    q_map = {q["id"]: q for q in json.load(f) if q.get("split") == "test"}

# Keywords or verification criteria for each question
verification_rules = {
    # Bảng (6)
    "adv_test_tab_01": ["28,672"],
    "adv_test_tab_02": ["1.0299"],
    "adv_test_tab_03": ["79.3", "79.5", "59.4"],
    "adv_test_tab_04": ["126", "16,384"],
    "adv_test_tab_05": ["cook", "stonemason", "athlete"],
    "adv_test_tab_06": ["1.4 trillion"],
    
    # Ảnh, sơ đồ (5)
    "adv_test_fig_01": ["can", "'t"],
    "adv_test_fig_02": ["unidirectional", "bidirectional", "left to right", "masked"],
    "adv_test_fig_03": ["application", "model", "infrastructure"],
    "adv_test_fig_04": ["comparison", "rank", "instructgpt", "interface"],
    "adv_test_fig_05": ["3/5", "5/5"],
    
    # Biểu đồ (5)
    "adv_test_chart_01": ["coding", "28%"],
    "adv_test_chart_02": ["one in three", "1/3", "200", "40%", "record high"],
    "adv_test_chart_03": ["internal", "risk", "privacy"],
    "adv_test_chart_04": ["documentation", "2x", "twice"],
    "adv_test_chart_05": ["400"],
    
    # Multi-hop (4)
    "adv_test_multi_01": ["128", "32", "4 times"],
    "adv_test_multi_02": ["45.88", "8", "7.7", "russian"],
    "adv_test_multi_03": ["application development"],
    "adv_test_multi_04": ["multiple", "verifier", "reward"],
    
    # Text (4)
    "adv_test_text_01": ["natural language supervision", "self-supervision"],
    "adv_test_text_02": ["13,500", "laion"],
    "adv_test_text_03": ["token", "unit"],
    "adv_test_text_04": ["inferred", "labels", "unsupervised"],
    
    # Thiếu bằng chứng (6)
    "adv_test_abs_01": ["không đủ bằng chứng"],
    "adv_test_abs_02": ["không đủ bằng chứng"],
    "adv_test_abs_03": ["không đủ bằng chứng"],
    "adv_test_abs_04": ["không đủ bằng chứng"],
    "adv_test_abs_05": ["không đủ bằng chứng"],
    "adv_test_abs_06": ["không đủ bằng chứng"]
}

audit_results = []
groups_stat = {}

for t in traces:
    cid = t["query_id"]
    group = t["main_group"]
    is_answerable = t["answerable"]
    status = t["status"]
    ans = t["answer"].lower()
    citations = t["citations"]
    q_obj = q_map[cid]

    if group not in groups_stat:
        groups_stat[group] = {"passed": 0, "total": 0, "errors": []}
    groups_stat[group]["total"] += 1

    passed = False
    reason = ""

    if is_answerable:
        if status != "answered":
            reason = f"Trạng thái {status} không đạt (yêu cầu answered)."
        elif not citations:
            reason = "Thiếu citation hợp lệ trích dẫn bằng chứng."
        else:
            rules = verification_rules.get(cid, [])
            match_count = sum(1 for r in rules if r.lower() in ans)
            if match_count >= 1:
                passed = True
                reason = f"Trả lời chính xác, trích dẫn {citations}, khớp {match_count}/{len(rules)} yếu tố kiểm chứng."
            else:
                reason = f"Chưa chứa đủ từ khóa kiểm chứng ({rules})."
    else:
        if status == "abstain":
            passed = True
            reason = "Nhận diện đúng sự thiếu hụt bằng chứng và từ chối an toàn."
        else:
            reason = f"Câu hỏi thiếu bằng chứng nhưng agent lại trả lời (status={status})."

    if passed:
        groups_stat[group]["passed"] += 1
    else:
        groups_stat[group]["errors"].append({"id": cid, "reason": reason})

    audit_results.append({
        "query_id": cid,
        "main_group": group,
        "answerable": is_answerable,
        "status": status,
        "passed": passed,
        "citations": citations,
        "answer": t["answer"],
        "reason": reason
    })

total_passed = sum(g["passed"] for g in groups_stat.values())
total_cases = sum(g["total"] for g in groups_stat.values())
pass_rate = (total_passed / total_cases) * 100

print(f"\n=======================================================")
print(f" ADVANCED BENCHMARK TEST AUDIT RESULTS")
print(f"=======================================================")
print(f"Total Passed: {total_passed} / {total_cases} ({pass_rate:.1f}%)\n")

for g, stat in groups_stat.items():
    g_pass = stat["passed"]
    g_tot = stat["total"]
    g_rate = (g_pass / g_tot) * 100
    print(f"- {g:18}: {g_pass}/{g_tot} ({g_rate:.1f}%)")

audit_file = ROOT_DIR / "results" / "advanced_test_audit.json"
with open(audit_file, "w", encoding="utf-8") as f:
    json.dump({
        "total_passed": total_passed,
        "total_cases": total_cases,
        "pass_rate": round(pass_rate, 2),
        "groups": groups_stat,
        "details": audit_results
    }, f, ensure_ascii=False, indent=2)

print(f"\nAudit saved to {audit_file}")
