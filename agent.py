# -*- coding: utf-8 -*-
"""
工具调用式 Agent：让模型自主决定检索轮次与翻页核对。

运行：
    python agent.py
"""
import json
import os
import time
from typing import Any, Dict, List

from openai import OpenAI

from rag_pipeline import rag_system

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

MAX_STEPS = 5          # 防止死循环
MAX_TOOL_CHARS = 2000  # 单次工具结果截断，避免上下文爆炸

# ---------------- 工具定义（给模型看的说明书）----------------
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "在课程资料库中语义检索相关片段。返回文件名、页码、相似度和内容摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索语句，尽量贴近资料里的表述"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 3"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "读取指定文件的指定页的完整内容，用于核对被截断或需要细节的信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "文件名，必须与 search_documents 返回的一致"},
                    "page": {"type": "integer", "description": "页码"},
                },
                "required": ["file", "page"],
            },
        },
    },
]

SYSTEM_PROMPT = """你是课程复习助手，可以调用工具查询课程资料。

规则：
1. 不确定资料里有没有相关内容时，先调用 search_documents。
2. 如果检索片段不完整（被截断、只提到"详见某页"、只答了一半）,调用 read_page 读取该页全文核对。
3. 最多调用 5 次工具。信息足够就立刻作答。
4. 同一轮不要用语义相近的 query 重复检索；单点事实类问题检索一次即可。
5. 只回答用户问到的内容，不要主动补充未被问及的细节（如答题要求、其他课程情况）。
6. 如果所有工具都无法提供答案，明确回答"根据已知资料无法回答该问题"，并在来源处写"来源：无"。
7. 回答用纯文本，不要使用 Markdown 语法，控制在 150 字以内。
8. 回答末尾单独一行标注来源，格式严格为：来源：文件名 第X页
9. 如果资料中没有直接对比用户所问的这两个概念，只回答"资料中没有直接对比 A 和 B 的内容"，不要再附加其他概念的对比内容。
"""


# ---------------- 工具实现 ----------------
def _search_documents(query: str, top_k: int = 3):
    """返回 (给模型看的文本, 原始 docs 列表)"""
    docs = rag_system.vector_store.search(query, top_k=top_k)
    if not docs:
        return "没有检索到相关内容。", []
    lines = []
    for i, d in enumerate(docs, 1):
        m = d.get("metadata", {})
        loc = m.get("page", m.get("slide", "-"))
        lines.append(
            f"[{i}] 文件：{m.get('filename')}  第{loc}页  "
            f"相似度：{d.get('score'):.3f}\n"
            f"内容：{(d.get('document') or '')[:300]}"
        )
    return "\n\n".join(lines), docs

def _read_page(file: str, page: int) -> str:
    """按文件名 + 页码取出该页所有块，拼回全文"""
    data = rag_system.vector_store.collection.get()
    metas = data.get("metadatas") or []
    docs = data.get("documents") or []
    rows = []
    for m, d in zip(metas, docs):
        if m.get("filename") != file:
            continue
        if str(m.get("page")) != str(page):
            continue
        rows.append((m.get("chunk_index", 0), d))
    if not rows:
        return f"未找到 {file} 第 {page} 页的内容。"
    rows.sort(key=lambda x: x[0])
    return "\n".join(d for _, d in rows)


def _dispatch(name: str, args: Dict[str, Any]):
    """返回 (给模型的文本, 命中的原始 docs)"""
    if name == "search_documents":
        return _search_documents(
            query=args["query"], top_k=int(args.get("top_k", 3))
        )
    if name == "read_page":
        return _read_page(file=args["file"], page=int(args["page"])), []
    return f"未知工具：{name}", []


# ---------------- Agent 主循环 ----------------
def agent_query(question: str, verbose: bool = True) -> Dict[str, Any]:
    t0 = time.time()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    seen = set()

    for step in range(1, MAX_STEPS + 1):
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            temperature=0.1,
        )
        msg = resp.choices[0].message

        # 模型决定不再调工具 → 结束
        if not msg.tool_calls:
            answer = (msg.content or "").strip()
            if verbose:
                print(f"  第 {step} 步：模型决定结束")
            return {
                "answer": answer,
                "steps": step,
                "trace": trace,
                "sources": sources,
                "elapsed_ms": int((time.time() - t0) * 1000),
            }

        # 把 assistant 的工具调用意图写回历史
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # 逐个执行工具
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            t_tool = time.time()
            found: List[Dict[str, Any]] = []
            try:
                result, found = _dispatch(name, args)
            except Exception as exc:  # noqa: BLE001
                result = f"工具执行失败：{type(exc).__name__}: {exc}"

            # 累积来源（按 文件+页+块 去重，保留最高分）
            for d in found:
                m = d.get("metadata", {})
                key = (m.get("filename"), m.get("page"), m.get("chunk_index"))
                if key in seen:
                    continue
                seen.add(key)
                sources.append({
                    "file": m.get("filename"),
                    "page": m.get("page", "-"),
                    "slide": m.get("slide", "-"),
                    "chunk_index": m.get("chunk_index", "-"),
                    "preview": (d.get("document") or "")[:40],
                    "score": d.get("score"),
                })

            trace.append({
                "step": step,
                "tool": name,
                "args": args,
                "elapsed_ms": int((time.time() - t_tool) * 1000),
                "result_chars": len(result),
            })
            if verbose:
                print(f"  第 {step} 步：调用 {name}({args}) "
                      f"→ 返回 {len(result)} 字")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result[:MAX_TOOL_CHARS],
            })

    # 超过最大步数：兜底
    return {
        "answer": "超过最大步数仍未得出结论，请把问题拆得更具体一些。",
        "steps": MAX_STEPS,
        "trace": trace,
        "sources": sources,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "exceeded": True,
    }


if __name__ == "__main__":
    tests = [
        "期末考几种题型？每种多少分？",                    # 单跳
        "哪门课的必考计算题里包含 IEEE754？",              # 多跳：需先定位再翻页
        "操作系统里进程调度有哪些算法？",                  # 应拒答
    ]
    for q in tests:
        print("\n" + "=" * 60)
        print("问：", q)
        res = agent_query(q)
        print("答：", res["answer"])
        print(f"步数：{res['steps']}   耗时：{res['elapsed_ms']} ms")
        print("轨迹：", [(t["tool"], t["args"]) for t in res["trace"]])