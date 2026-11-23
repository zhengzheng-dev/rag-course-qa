# -*- coding: utf-8 -*-
"""
Agent 版评估脚本：在 22 题评估集上同时衡量【效果】与【效率】。

用法（在任意目录执行均可）：
    python eval/run_agent_eval.py

可选环境变量：
    EVAL_NOTE   写入 agent_runs.jsonl 的备注，用于区分实验
"""
import contextlib
import datetime
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import agent_query                       # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.json"
REPORT_DIR = ROOT / "eval" / "reports"

REFUSAL_MARKS = ("无法回答", "无法确定", "没有找到", "未提及",
                 "不包含", "不知道", "无法从资料", "来源：无",
                 "没有直接对比", "未给出", "没有给出", "未说明")

# ---- 判定阈值（与固定流水线版本保持一致）----
MIN_TOP1_SCORE = 0.52
MAX_ANSWER_CHARS = 250
MAX_RETRIES = 3


# ---------------- 基础工具 ----------------
def load_questions():
    return json.loads(QUESTIONS.read_text(encoding="utf-8"))


def source_hit(retrieved, expect):
    if not expect:
        return None
    for exp in expect:
        for got in retrieved:
            if exp["file"] not in (got.get("file") or ""):
                continue
            ok = True
            if "page" in exp:
                ok = str(got.get("page")) == str(exp["page"])
            if "slide" in exp:
                ok = str(got.get("slide")) == str(exp["slide"])
            if ok:
                return True
    return False


def keyword_coverage(answer, keywords):
    if not keywords:
        return None
    hit = sum(1 for k in keywords if k.lower() in answer.lower())
    return hit / len(keywords)


def is_refusal(answer):
    return any(m in answer for m in REFUSAL_MARKS)


def strip_source_line(answer: str) -> str:
    """长度红线只统计正文；结尾的“来源：xxx”属于格式要求，不计入字数"""
    lines = [ln for ln in answer.splitlines() if not ln.strip().startswith("来源：")]
    return "\n".join(lines)


def agent_with_retry(question, retries=MAX_RETRIES):
    last = None
    for attempt in range(1, retries + 1):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return agent_query(question, verbose=False)
        except Exception as exc:                       # noqa: BLE001
            last = exc
            if attempt == retries:
                break
            wait = 2 ** attempt
            print(f"    [重试 {attempt}/{retries}] {type(exc).__name__}，{wait}s 后重试")
            time.sleep(wait)
    raise last


def save_partial(results):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "agent_partial.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------- 主流程 ----------------
def run():
    qs = load_questions()
    print(f"共 {len(qs)} 道题，开始 Agent 评估...\n")

    results = []
    for idx, q in enumerate(qs, start=1):
        print(f"  [{idx}/{len(qs)}] {q['id']}  {q['question']}")
        try:
            res = agent_with_retry(q["question"])
        except Exception as exc:                       # noqa: BLE001
            print(f"    ✗ 跳过（{type(exc).__name__}: {exc}）")
            results.append({
                "id": q["id"], "category": q["category"],
                "question": q["question"], "answer": "",
                "answerable": q["answerable"], "source_hit": None,
                "keyword_coverage": None, "refused": False,
                "top1_score": 0.0, "rounds": 0, "tool_calls": 0,
                "tools_used": [], "exceeded": False,
                "elapsed_ms": 0, "char_count": 0, "length_ok": False,
                "passed": False, "sources": [],
                "error": f"{type(exc).__name__}: {exc}",
            })
            save_partial(results)
            continue

        answer = res.get("answer", "") or ""
        sources = res.get("sources", []) or []
        trace = res.get("trace", []) or []

        hit = source_hit(sources, q.get("expect_sources", []))
        cov = keyword_coverage(answer, q.get("expect_keywords", []))
        refused = is_refusal(answer)

        scores = [s.get("score") for s in sources if s.get("score") is not None]
        top1_score = max(scores) if scores else 0.0

        body = strip_source_line(answer)
        char_count = len(body.replace("\n", "").replace(" ", ""))
        max_chars = q.get("max_chars", MAX_ANSWER_CHARS)
        length_ok = char_count <= max_chars

        if q["answerable"]:
            min_cov = q.get("min_coverage", 0.5)
            cov_ok = cov is not None and cov >= min_cov
            passed = cov_ok and (hit is not False) and length_ok
        else:
            passed = refused

        tools_used = [t.get("tool") for t in trace]
        results.append({
            "id": q["id"], "category": q["category"],
            "question": q["question"], "answer": answer,
            "answerable": q["answerable"],
            "source_hit": hit, "keyword_coverage": cov, "refused": refused,
            "top1_score": top1_score,
            "rounds": res.get("steps", 0),                       # LLM 轮次
            "tool_calls": len(trace),                            # 工具调用次数
            "tools_used": tools_used,
            "trace":trace,
            "exceeded": bool(res.get("exceeded")),
            "elapsed_ms": res.get("elapsed_ms", 0),
            "char_count": char_count, "length_ok": length_ok,
            "passed": passed, "sources": sources,
        })
        save_partial(results)

        print(f"    轮次={res.get('steps')} 工具={len(trace)} "
              f"耗时={res.get('elapsed_ms')}ms 通过={passed}")

    report(results)


def pctl(values, p):
    if not values:
        return 0
    data = sorted(values)
    idx = min(len(data) - 1, int(round((p / 100) * (len(data) - 1))))
    return data[idx]


def report(results):
    n = len(results)
    errors = [r for r in results if r.get("error")]
    ok = [r for r in results if not r.get("error")]

    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]

    # ---- 效果指标 ----
    hits = [r["source_hit"] for r in answerable if r["source_hit"] is not None]
    hit_rate = sum(hits) / len(hits) if hits else 0
    covs = [r["keyword_coverage"] for r in answerable
            if r["keyword_coverage"] is not None]
    avg_cov = statistics.mean(covs) if covs else 0
    refused_ok = sum(1 for r in unanswerable if r["refused"])
    len_ok_n = sum(1 for r in ok if r["length_ok"])
    passed = sum(1 for r in results if r["passed"])

    # ---- 效率指标 ----
    rounds = [r["rounds"] for r in ok]
    tools = [r["tool_calls"] for r in ok]
    elapsed = [r["elapsed_ms"] for r in ok]
    exceeded_n = sum(1 for r in ok if r["exceeded"])

    avg_rounds = statistics.mean(rounds) if rounds else 0
    avg_tools = statistics.mean(tools) if tools else 0
    avg_ms = statistics.mean(elapsed) if elapsed else 0
    p95_ms = pctl(elapsed, 95)
    exceeded_ratio = exceeded_n / len(ok) if ok else 0

    tool_dist = {}
    for r in ok:
        for t in r["tools_used"]:
            tool_dist[t] = tool_dist.get(t, 0) + 1

    print("\n" + "=" * 64)
    print(f"总题数                 : {n}")
    print(f"API 失败题数           : {len(errors)}")
    for e in errors:
        print(f"   - [{e['id']}] {e['error'][:70]}")
    print("—— 效果指标 ——")
    print(f"答案通过率             : {passed}/{n} = {passed / n:.1%}")
    print(f"检索命中率 (来源级)    : {hit_rate:.1%}   ({sum(hits)}/{len(hits)})")
    print(f"关键词覆盖率 (均值)    : {avg_cov:.1%}")
    print(f"答案长度合规 (≤{MAX_ANSWER_CHARS}字) : {len_ok_n}/{len(ok)}")
    if unanswerable:
        print(f"拒答正确率             : {refused_ok / len(unanswerable):.1%}"
              f"   ({refused_ok}/{len(unanswerable)})")
    print("—— 效率指标 ——")
    print(f"平均轮次               : {avg_rounds:.2f}")
    print(f"平均工具调用数         : {avg_tools:.2f}")
    print(f"超步数比例             : {exceeded_ratio:.1%}   ({exceeded_n}/{len(ok)})")
    print(f"平均耗时               : {avg_ms:.0f} ms")
    print(f"P95 耗时               : {p95_ms:.0f} ms")
    if tool_dist:
        print("—— 工具使用分布 ——")
        for name, cnt in sorted(tool_dist.items(), key=lambda x: -x[1]):
            print(f"  {name:<20} : {cnt} 次")
    print("=" * 64)

    print("\n按题型看（通过率 / 平均轮次 / 平均工具调用）:")
    cats = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    for cat, rs in cats.items():
        p = sum(1 for r in rs if r["passed"])
        avg_r = statistics.mean([x["rounds"] for x in rs]) if rs else 0
        avg_t = statistics.mean([x["tool_calls"] for x in rs]) if rs else 0
        print(f"  {cat:<16} {p}/{len(rs)}   轮次 {avg_r:.2f}   工具 {avg_t:.2f}")

    fails = [r for r in results if not r["passed"]]
    if fails:
        print(f"\n未通过的题（{len(fails)} 道）：")
        for r in fails:
            print(f"\n  [{r['id']}] {r['question']}")
            print(f"    题型      : {r['category']}")
            if r.get("error"):
                print(f"    API 错误  : {r['error'][:80]}")
                continue
            print(f"    来源命中  : {r['source_hit']}")
            print(f"    关键词覆盖: {r['keyword_coverage']}")
            print(f"    字数      : {r['char_count']}  (合规: {r['length_ok']})")
            print(f"    是否拒答  : {r['refused']}")
            print(f"    轮次/工具 : {r['rounds']} / {r['tool_calls']}  {r['tools_used']}")
            print(f"    答案      : {(r['answer'] or '')[:120]}")

    # ---- 落盘 ----
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "agent_latest.json"
    out.write_text(json.dumps({
        "summary": {
            "total": n, "errors": len(errors), "passed": passed,
            "effect": {
                "hit_rate": hit_rate,
                "avg_keyword_coverage": avg_cov,
                "refusal_accuracy": (refused_ok / len(unanswerable)
                                     if unanswerable else 0),
                "length_ok": len_ok_n,
            },
            "efficiency": {
                "avg_rounds": avg_rounds,
                "avg_tool_calls": avg_tools,
                "exceeded_ratio": exceeded_ratio,
                "avg_elapsed_ms": avg_ms,
                "p95_elapsed_ms": p95_ms,
                "tool_distribution": tool_dist,
            },
        },
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    note = os.environ.get("EVAL_NOTE", f"{n}题 Agent 基线")
    with (REPORT_DIR / "agent_runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "total": n, "errors": len(errors), "passed": passed,
            "hit_rate": round(hit_rate, 4),
            "avg_cov": round(avg_cov, 4),
            "avg_rounds": round(avg_rounds, 3),
            "avg_tools": round(avg_tools, 3),
            "exceeded_ratio": round(exceeded_ratio, 4),
            "avg_ms": round(avg_ms, 1),
            "p95_ms": round(p95_ms, 1),
            "note": note,
        }, ensure_ascii=False) + "\n")

    print(f"\n报告已保存: {out}")
    print(f"历史已追加: {REPORT_DIR / 'agent_runs.jsonl'}   note={note}")


if __name__ == "__main__":
    run()