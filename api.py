# -*- coding: utf-8 -*-
"""
FastAPI 服务层：把 RAG 能力暴露为 HTTP 接口。

启动：
    uvicorn api:app --host 127.0.0.1 --port 8000
接口文档：
    http://127.0.0.1:8000/docs
"""
import re
import logging
import time
from typing import List, Optional
from fastapi.responses import JSONResponse, PlainTextResponse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_pipeline import rag_system

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rag.api")

app = FastAPI(
    title="课程复习问答助手 API",
    version="1.0.0",
    description="基于 RAG 的课程资料问答服务：答案带页码来源，超出资料范围时显式拒答。",
)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    """兜底：任何未捕获异常都返回结构化 JSON，而不是裸 500"""
    log.exception("未处理异常: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误: {type(exc).__name__}"},
    )

# 允许本地前端 / Streamlit 跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_RETRIES = 3          # 与评测脚本一致：2s / 4s / 8s 指数退避


# ---------------- 请求 / 响应模型 ----------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=200, description="用户问题")
    top_k: int = Field(3, ge=1, le=10, description="检索条数，默认 3")


class SourceItem(BaseModel):
    file: str
    page: Optional[int] = None
    slide: Optional[int] = None
    chunk_index: Optional[int] = None
    score: Optional[float] = None
    preview: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    elapsed_ms: int


# ---------------- 内部工具 ----------------
def _query_with_retry(question: str, top_k: int):
    """调用 RAG；服务端抖动时指数退避重试"""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return rag_system.query(question, top_k=top_k)
        except Exception as exc:                      # noqa: BLE001
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            wait = 2 ** attempt
            log.warning("第 %d 次调用失败(%s)，%ds 后重试", attempt,
                        type(exc).__name__, wait)
            time.sleep(wait)
    raise last_exc


def _int_or_none(v):
    """把 '-'、''、None 这类占位符统一转成 None；能转 int 的转 int"""
    if v is None or v == "-" or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_source(s: dict) -> SourceItem:
    return SourceItem(
        file=s.get("file", ""),
        page=_int_or_none(s.get("page")),
        slide=_int_or_none(s.get("slide")),
        chunk_index=_int_or_none(s.get("chunk_index")),
        score=round(float(s["score"]), 4) if s.get("score") is not None else None,
        preview=(s.get("preview") or "")[:120],
    )

_MD_RULES = [
    (re.compile(r"\*\*(.+?)\*\*", re.S), r"\1"),   # **粗体** -> 粗体
    (re.compile(r"`([^`]*)`"), r"\1"),             # `代码`   -> 代码
    (re.compile(r"(?<!\*)\*(?!\*)"), ""),          # 残留的孤立 *
]

_CITE_RULES = [
    # （依据：参考资料[1]） / （参考资料[1]） / (来源：[1])
    re.compile(r"[（(]\s*(依据|来源|参考)[:：]?\s*参考资料?\s*\[?\d+\]?\s*[)）]"),
    re.compile(r"[（(]\s*参考资料?\s*\[?\d+\]\s*[)）]"),
    # 行首的“根据参考资料[1]，”
    re.compile(r"^\s*根据参考资料?\s*\[?\d+\]?\s*[，,：:]?\s*", re.M),
]


def _clean_text(s: str) -> str:
    """把模型输出整理成适合纯文本展示的内容：
    去掉 Markdown 标记，去掉读者无法对应的编号引用。"""
    if not s:
        return ""
    for pat, rep in _MD_RULES:
        s = pat.sub(rep, s)
    for pat in _CITE_RULES:
        s = pat.sub("", s)
    s = re.sub(r"[ \t]+\n", "\n", s)      # 行尾空格
    s = re.sub(r"\n{3,}", "\n\n", s)      # 多余空行
    return s.strip()

# ---------------- 接口 ----------------
@app.get("/health", summary="健康检查")
def health():
    """返回服务状态、已索引块数与当前配置。"""
    try:
        chunks = rag_system.vector_store.collection.count()
    except Exception:                                  # noqa: BLE001
        chunks = -1
    return {
        "status": "ok",
        "indexed_chunks": chunks,
        "llm": "deepseek-chat",
        "embedding": "BAAI/bge-m3",
    }

@app.get("/ask", response_class=PlainTextResponse, summary="浏览器直接提问（纯文本）")
def ask(q: str, top_k: int = 3):
    """
    直接在浏览器地址栏提问，返回纯文本答案 + 来源。
    例：http://127.0.0.1:8000/ask?q=TCP三次握手的过程
    """
    try:
        res = _query_with_retry(q.strip(), top_k)
    except Exception as exc:                           # noqa: BLE001
        log.exception("ask 失败")
        raise HTTPException(status_code=503, detail="模型服务暂时不可用") from exc

    answer = _clean_text(res.get("answer", ""))
    lines = [answer, "", "—— 来源 ——"]
    for s in res.get("sources", []):
        loc = s.get("page")
        unit = "页"
        if loc in (None, "-"):
            loc, unit = s.get("slide"), "张"
        sc = s.get("score")
        sc_txt = f"{sc:.3f}" if isinstance(sc, (int, float)) else "-"
        lines.append(f"· {s.get('file')}  第 {loc} {unit}  score={sc_txt}")
    return "\n".join(lines)

@app.post("/query", response_model=QueryResponse, summary="提问")
def query(req: QueryRequest):
    """
    提问并返回答案 + 来源。

    注意：这里用 `def`（同步）而不是 `async def`。
    RAG 内部是阻塞的网络调用（嵌入 + 生成），同步函数会被 FastAPI
    放到线程池执行，不会阻塞事件循环；写成 async def 反而会卡住整个服务。
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空")

    t0 = time.time()
    try:
        res = _query_with_retry(question, req.top_k)
    except Exception as exc:                           # noqa: BLE001
        log.exception("查询失败")
        raise HTTPException(
            status_code=503,
            detail="模型服务暂时不可用，请稍后重试",
        ) from exc

    elapsed = int((time.time() - t0) * 1000)
    log.info("question=%r top_k=%d elapsed=%dms", question, req.top_k, elapsed)

    return QueryResponse(
        answer=res.get("answer", ""),
        sources=[_to_source(s) for s in res.get("sources", [])],
        elapsed_ms=elapsed,
    )