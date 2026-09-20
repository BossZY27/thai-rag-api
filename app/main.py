import hashlib
import json
import math
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import pdfplumber
import requests
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"
DOCUMENT_DIR = DATA_DIR / "documents"
DB_PATH = DATA_DIR / "rag.sqlite3"
STATIC_DIR = APP_DIR / "static"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5:9b-q4_K_M")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
API_KEY = os.getenv("RAG_API_KEY", "")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "5200"))
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "360"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
UNKNOWN_ANSWER = "ยังไม่พบข้อมูลที่ตอบคำถามนี้โดยตรงในฐานข้อมูล"

app = FastAPI(title="Thai RAG API", version="0.3.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

CATEGORY_LABELS = {
    "overview": "ภาพรวมประกาศ",
    "definitions": "คำจำกัดความ",
    "fund_types": "ประเภททุน",
    "amounts": "วงเงินและอัตราทุน",
    "rules": "หลักเกณฑ์และเงื่อนไข",
    "process": "ขั้นตอนการขอทุน",
    "publication_rewards": "รางวัลผลงานตีพิมพ์",
    "presentation": "ทุนนำเสนอผลงาน",
    "other": "ข้อมูลทั่วไป",
}


class ChatRequest(BaseModel):
    question: str
    top_k: int = 8
    category: str | None = None
    mode: str = "fast"


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    elapsed_ms: int
    log_id: int | None = None
    intent: str | None = None
    debug: dict[str, Any] | None = None


class KnowledgeTextRequest(BaseModel):
    title: str
    text: str
    source_ref: str | None = None
    page: int = 1
    category: str | None = None


class FeedbackRequest(BaseModel):
    log_id: int | None = None
    question: str
    answer: str
    rating: str
    note: str | None = None


class BatchTestRequest(BaseModel):
    questions: list[str]
    mode: str = "fast"
    category: str | None = None


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def text_has_any(text: str, terms: list[str]) -> bool:
    compact = normalize_text(text)
    return any(normalize_text(term) in compact for term in terms)


def classify_chunk(text: str, page: int, forced_category: str | None = None) -> dict[str, Any]:
    if forced_category and forced_category in CATEGORY_LABELS:
        return {
            "category": forced_category,
            "subcategory": CATEGORY_LABELS[forced_category],
            "level": 1,
            "tags": [forced_category],
        }

    if text_has_any(text, ["หมายความว่า", "“บุคลากร”", "“ผู้ขอรับทุน”", "“ผู้ได้รับทุน”"]):
        matched = [term for term in ["หมายความว่า", "บุคลากร", "ผู้ขอรับทุน", "ผู้ได้รับทุน"] if text_has_any(text, [term])]
        return {
            "category": "definitions",
            "subcategory": "คำจำกัดความ",
            "level": 1,
            "tags": matched[:6],
        }

    checks = [
        ("presentation", "ทุนนำเสนอผลงาน", 2, ["นำเสนอ", "ประชุมวิชาการ", "โปสเตอร์", "บรรยาย"]),
        ("publication_rewards", "รางวัลผลงานตีพิมพ์", 2, ["ตีพิมพ์", "วารสาร", "SCOPUS", "ISI", "WOS", "Q1", "Q2"]),
        ("amounts", "วงเงินและอัตราทุน", 1, ["อัตราการให้เงินทุน", "วงเงินรวม", "ไม่เกิน", "บาท", "จำนวนทุน"]),
        ("fund_types", "ประเภททุน", 1, ["ประเภททุน", "ทุนส่งเสริม", "ทุนอุดหนุน", "ทุนวิจัย", "ทุนนวัตกรรม"]),
        ("process", "ขั้นตอนการขอทุน", 2, ["เสนอขอรับทุน", "ขอรับทุน", "ยื่น", "อนุมัติ", "คณะกรรมการ"]),
        ("rules", "หลักเกณฑ์และเงื่อนไข", 1, ["หลักเกณฑ์", "เงื่อนไข", "ภาระผูกพัน", "บังคับ", "ประกาศ"]),
    ]
    for category, subcategory, level, terms in checks:
        if text_has_any(text, terms):
            matched = [term for term in terms if text_has_any(text, [term])]
            return {
                "category": category,
                "subcategory": subcategory,
                "level": level,
                "tags": matched[:6],
            }

    return {
        "category": "overview" if page <= 1 else "other",
        "subcategory": "ภาพรวมประกาศ" if page <= 1 else "ข้อมูลทั่วไป",
        "level": 2 if page <= 1 else 3,
        "tags": ["overview"] if page <= 1 else [],
    }


def backfill_categories(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT id, document_id, page, text
        FROM chunks
        """
    ).fetchall()
    for row in rows:
        meta = classify_chunk(row["text"], row["page"])
        conn.execute(
            """
            UPDATE chunks
            SET category = ?, subcategory = ?, level = ?, tags = ?
            WHERE id = ?
            """,
            (
                meta["category"],
                meta["subcategory"],
                meta["level"],
                json.dumps(meta["tags"], ensure_ascii=False),
                row["id"],
            ),
        )

    document_ids = conn.execute("SELECT id FROM documents").fetchall()
    for document in document_ids:
        top = conn.execute(
            """
            SELECT category, COUNT(*) AS count
            FROM chunks
            WHERE document_id = ?
            GROUP BY category
            ORDER BY count DESC
            LIMIT 1
            """,
            (document["id"],),
        ).fetchone()
        if top:
            conn.execute(
                "UPDATE documents SET category = ? WHERE id = ?",
                (top["category"], document["id"]),
            )


def init_db() -> None:
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_ref TEXT,
                category TEXT NOT NULL DEFAULT 'other',
                pages_with_text INTEGER NOT NULL DEFAULT 1,
                chunks INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL,
                page INTEGER NOT NULL,
                category TEXT NOT NULL DEFAULT 'other',
                subcategory TEXT NOT NULL DEFAULT '',
                level INTEGER NOT NULL DEFAULT 3,
                tags TEXT NOT NULL DEFAULT '[]',
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
            """
        )
        ensure_column(conn, "documents", "category", "TEXT NOT NULL DEFAULT 'other'")
        ensure_column(conn, "chunks", "category", "TEXT NOT NULL DEFAULT 'other'")
        ensure_column(conn, "chunks", "subcategory", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "chunks", "level", "INTEGER NOT NULL DEFAULT 3")
        ensure_column(conn, "chunks", "tags", "TEXT NOT NULL DEFAULT '[]'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_category ON chunks(category)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS qa_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                mode TEXT NOT NULL,
                category TEXT,
                intent TEXT,
                elapsed_ms INTEGER NOT NULL,
                sources_json TEXT NOT NULL,
                debug_json TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_id INTEGER,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                rating TEXT NOT NULL,
                note TEXT,
                created_at INTEGER NOT NULL
            )
            """
        )
        backfill_categories(conn)


@app.on_event("startup")
def startup() -> None:
    init_db()


def ollama_embed(text: str) -> list[float]:
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text, "keep_alive": OLLAMA_KEEP_ALIVE},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot reach Ollama embedding model: {exc}",
        ) from exc

    embedding = payload.get("embedding")
    if not embedding:
        raise HTTPException(status_code=502, detail="Ollama returned no embedding")
    return embedding


def build_context(context_blocks: list[dict[str, Any]]) -> str:
    parts = []
    used_chars = 0
    for block in context_blocks:
        text = block["text"]
        remaining = MAX_CONTEXT_CHARS - used_chars
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining].rsplit(" ", 1)[0] or text[:remaining]
        parts.append(f"[source: {block['title']} page {block['page']}]\n{text}")
        used_chars += len(text)
    return "\n\n".join(parts)


def ollama_chat(question: str, context_blocks: list[dict[str, Any]]) -> str:
    system = (
        "You are a Thai-language document QA assistant. "
        "Answer in Thai only. Use only the provided context. "
        "Do not guess, infer, or answer from merely related text. "
        "If the provided context does not directly answer the question, say exactly: ยังไม่พบข้อมูลที่ตอบคำถามนี้โดยตรงในฐานข้อมูล. "
        "Keep the answer concise, practical, and use bullet points only when it helps."
    )
    user = f"Context from database:\n{build_context(context_blocks)}\n\nQuestion: {question}"

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "num_ctx": 4096,
                    "num_predict": MAX_ANSWER_TOKENS,
                    "repeat_penalty": 1.08,
                },
                "think": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
            },
            timeout=240,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot reach Ollama chat model: {exc}",
        ) from exc

    message = payload.get("message", {})
    content = message.get("content", "").strip()
    if content:
        return content
    if message.get("thinking"):
        return "โมเดลใช้เวลาคิดแต่ไม่ได้สร้างคำตอบสุดท้าย ลองถามให้เฉพาะเจาะจงขึ้น หรือใช้โมเดลที่ปิด thinking ได้ดีกว่า"
    return ""


def extract_amounts(text: str) -> list[str]:
    amounts = re.findall(r"(?:ไม่เกิน\s*)?[\d,]+\s*บาท", text)
    seen = []
    for amount in amounts:
        amount = re.sub(r"\s+", " ", amount.strip())
        if amount not in seen:
            seen.append(amount)
    return seen[:8]


INTENT_CATEGORY_HINTS = {
    "source": ["overview", "fund_types", "definitions"],
    "agency": ["fund_types", "overview", "definitions"],
    "eligibility": ["definitions", "rules", "process"],
    "application_requirements": ["process", "rules", "definitions"],
    "amounts": ["amounts", "presentation", "publication_rewards", "fund_types"],
    "presentation": ["presentation", "amounts"],
    "publication_rewards": ["publication_rewards", "amounts"],
    "fund_types": ["fund_types", "amounts"],
    "overview": ["overview", "fund_types", "definitions"],
}


def has_any_compact(text: str, terms: list[str]) -> bool:
    compact = normalize_text(text)
    return any(normalize_text(term) in compact for term in terms)


def detect_question_intent(question: str) -> str:
    compact = normalize_text(question)
    checks = [
        ("source", ["เอาข้อมูลมาจากไหน", "ข้อมูลจากไหน", "อ้างอิงจากไหน", "มาจากหน้าไหน", "จากหน้าไหน", "หน้าไหน", "ที่มา", "แหล่งข้อมูล"]),
        ("agency", ["หน่วยงานไหน", "หน่วยงานอะไร", "ของหน่วยงาน", "เป็นของใคร", "ใครออกประกาศ", "ออกโดยใคร", "สังกัดไหน"]),
        ("eligibility", ["สมัครได้", "ใครสมัคร", "ผู้มีสิทธิ", "คุณสมบัติ", "คนธรรมดา", "พระ", "คณะธรรมศาสตร์"]),
        ("application_requirements", ["สมัคร", "เตรียมอะไร", "เอกสารอะไร", "ต้องเตรียม", "ต้องรอ", "รออะไร", "ต้องขอ", "ขออะไร", "ก่อนสมัคร", "ก่อนยื่น", "ขั้นตอน", "ยื่นยังไง", "ขอทุนยังไง"]),
        ("amounts", ["วงเงิน", "กี่บาท", "เท่าไหร่", "เท่าไร", "ได้เงิน", "อัตราทุน", "จำนวนทุน"]),
        ("presentation", ["นำเสนอ", "ประชุม", "โปสเตอร์", "บรรยาย"]),
        ("publication_rewards", ["ตีพิมพ์", "วารสาร", "scopus", "isi", "wos", "รางวัล"]),
        ("fund_types", ["ประเภททุน", "มีทุนอะไร", "ทุนอะไรบ้าง"]),
        ("overview", ["เกี่ยวกับอะไร", "ประกาศนี้คือ", "ประกาศอะไร", "สรุป"]),
    ]
    for intent, terms in checks:
        if any(normalize_text(term) in compact for term in terms):
            return intent
    return "general"


def is_source_follow_up(question: str) -> bool:
    compact = normalize_text(question)
    source_terms = ["หน้าไหน", "จากหน้าไหน", "มาจากไหน", "ข้อมูลจากไหน", "เอามาจากไหน", "เอาข้อมูลมาจากไหน", "ไปเอาข้อมูล", "อ้างอิง", "แหล่งข้อมูล", "ที่มา"]
    follow_up_terms = ["ข้อมูลนี้", "คำตอบนี้", "อันนี้", "ตรงนี้", "เมื่อกี้", "ที่ตอบ", "คุณ", "ai", "ระบบ"]
    return any(normalize_text(term) in compact for term in source_terms) and any(normalize_text(term) in compact for term in follow_up_terms)


def answer_from_previous_sources(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "คำตอบก่อนหน้าไม่มีแหล่งอ้างอิง เพราะระบบประเมินว่ายังไม่พบข้อมูลที่ตอบคำถามนั้นโดยตรงในฐานข้อมูล"

    seen = set()
    lines = []
    for source in sources:
        category = CATEGORY_LABELS.get(source.get("category"), source.get("subcategory") or source.get("category") or "ไม่ระบุหมวด")
        label = f"{source.get('title', 'ไม่ระบุเอกสาร')} หน้า {source.get('page', '-')} · {category}"
        if label in seen:
            continue
        seen.add(label)
        lines.append(f"- {label}")
    return "คำตอบก่อนหน้าอ้างอิงจาก:\n" + "\n".join(lines)


def evidence_window(text: str, terms: list[str], limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    compact = normalize_text(text)
    positions = []
    for term in terms:
        index = compact.find(normalize_text(term))
        if index >= 0:
            positions.append(index)
    if not positions:
        return clean_snippet(text, limit)
    rough_index = max(0, min(positions) - 90)
    window = text[rough_index : rough_index + limit].strip()
    return clean_snippet(window, limit)


def unique_source_lines(chunks: list[dict[str, Any]], limit: int = 5) -> list[str]:
    seen = set()
    lines = []
    for chunk in chunks:
        key = (chunk["title"], chunk["page"], chunk["subcategory"])
        if key in seen:
            continue
        seen.add(key)
        label = chunk.get("subcategory") or CATEGORY_LABELS.get(chunk.get("category"), chunk.get("category", "ไม่ระบุหมวด"))
        lines.append(f"- {chunk['title']} หน้า {chunk['page']} ({label})")
        if len(lines) >= limit:
            break
    return lines


def find_evidence_chunk(chunks: list[dict[str, Any]], terms: list[str], fallback: bool = False) -> dict[str, Any] | None:
    for chunk in chunks:
        if has_any_compact(chunk["text"], terms):
            return chunk
    return chunks[0] if fallback and chunks else None


def best_keyword_score(chunks: list[dict[str, Any]]) -> float:
    return max((float(chunk.get("keyword_score", 0.0)) for chunk in chunks), default=0.0)


def extract_definition(text: str, term: str) -> str | None:
    clean = re.sub(r"\s+", " ", text).strip()
    pattern = rf"[“\"]{re.escape(term)}[”\"]\s*หมายความว่า\s*(.*?)(?=\s*[“\"]|ข้อ\s*\d+|$)"
    match = re.search(pattern, clean)
    if not match:
        return None
    return clean_snippet(match.group(1), 220)


def eligibility_definitions(chunks: list[dict[str, Any]]) -> list[str]:
    definitions = []
    seen = set()
    for term in ["บุคลากร", "ผู้ขอรับทุน", "ผู้ได้รับทุน"]:
        for chunk in chunks:
            definition = extract_definition(chunk["text"], term)
            if definition and term not in seen:
                definitions.append(f"- {term}: {definition}")
                seen.add(term)
                break
    return definitions


def detect_agency_from_chunks(chunks: list[dict[str, Any]]) -> dict[str, str] | None:
    combined = " ".join(re.sub(r"\s+", " ", chunk["text"]) for chunk in chunks[:6])
    if not has_any_compact(combined, ["มหาวิทยาลัยขอนแก่น", "วิทยาลัยการคอมพิวเตอร์"]):
        return None
    issuer = "มหาวิทยาลัยขอนแก่น" if has_any_compact(combined, ["ประกาศมหาวิทยาลัยขอนแก่น"]) else ""
    subject_owner = ""
    if has_any_compact(combined, ["วิทยาลัยการคอมพิวเตอร์"]):
        subject_owner = "วิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น"
    return {
        "issuer": issuer,
        "subject_owner": subject_owner,
    }


def extract_funding_items(chunks: list[dict[str, Any]], limit: int = 7) -> list[str]:
    items = []
    seen = set()
    for chunk in chunks:
        text = re.sub(r"\s+", " ", chunk["text"]).strip()
        segments = re.split(r"(?=(?:\d+(?:\.\d+)+|ข้อ\s*\d+)\s+)", text)
        if len(segments) <= 1:
            segments = re.split(r"(?=ไม่เกิน\s*[\d,]+\s*บาท)", text)
        for segment in segments:
            segment = segment.strip(" -")
            if re.match(r"^[\d,]+\s*บาท", segment):
                continue
            if "บาท" not in segment:
                continue
            if not has_any_compact(segment, ["ทุน", "รางวัล", "ค่าตอบแทน", "สนับสนุน", "วงเงิน", "อัตรา"]):
                continue
            snippet = clean_snippet(segment, 230)
            key = normalize_text(snippet)
            if key in seen:
                continue
            seen.add(key)
            items.append(f"- {snippet}")
            if len(items) >= limit:
                return items
    return items


def clean_snippet(text: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^-?\d+-\s*", "", text)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def fast_answer(question: str, chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return UNKNOWN_ANSWER

    intent = detect_question_intent(question)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        by_category.setdefault(chunk["category"], []).append(chunk)

    if intent == "general" and best_keyword_score(chunks) < 0.18:
        return UNKNOWN_ANSWER

    if intent == "source":
        lines = ["ผมตอบจากฐานความรู้ที่ถูกบันทึกไว้ในระบบตอนนี้ โดยคำถามนี้ดึงข้อมูลจาก:"]
        lines.extend(unique_source_lines(chunks))
        return "\n".join(lines)

    if intent == "agency":
        agency = detect_agency_from_chunks(chunks)
        if not agency:
            return UNKNOWN_ANSWER
        lines = []
        if agency["issuer"]:
            lines.append(f"ผู้ออกประกาศ: {agency['issuer']}")
        if agency["subject_owner"]:
            lines.append(f"หน่วยงานเจ้าของเรื่อง/กองทุนที่กล่าวถึง: {agency['subject_owner']}")
        if not lines:
            return UNKNOWN_ANSWER
        return "\n".join(lines)

    if intent == "eligibility":
        terms = ["บุคลากร", "สังกัด", "คุณสมบัติ", "ผู้รับทุน", "ข้าราชการ", "พนักงานมหาวิทยาลัย", "ลูกจ้าง"]
        evidence = find_evidence_chunk(chunks, terms)
        if not evidence:
            return UNKNOWN_ANSWER
        definitions = eligibility_definitions(chunks)
        snippet = "\n".join(definitions) if definitions else f"- {evidence_window(evidence['text'], terms, 420)}"
        question_terms = [term for term in ["พระ", "คนธรรมดา", "คณะธรรมศาสตร์"] if has_any_compact(question, [term])]
        if question_terms and not has_any_compact(evidence["text"], question_terms):
            return (
                f"{UNKNOWN_ANSWER}\n"
                f"- ยังไม่พบข้อความที่ระบุว่า “{question_terms[0]}” สมัครได้โดยตรง"
            )
        return f"จากข้อมูลที่พบ เงื่อนไขผู้เกี่ยวข้อง/ผู้มีสิทธิ์สมัครมีใจความว่า:\n{snippet}"

    if intent == "application_requirements":
        terms = ["เอกสารสมัคร", "เอกสารประกอบ", "แบบฟอร์ม", "เสนอขอรับทุน", "ขั้นตอนการขอทุน", "ขอรับทุน"]
        evidence = find_evidence_chunk(chunks, terms)
        if not evidence:
            return (
                f"{UNKNOWN_ANSWER}\n"
                "- ฐานข้อมูลตอนนี้มีหลักเกณฑ์/วงเงินบางส่วน แต่ยังไม่พบรายการเอกสารสมัคร ขั้นตอนสมัคร หรือสิ่งที่ต้องรอแบบชัดเจน"
            )
        snippet = evidence_window(evidence["text"], terms, 420)
        asks_documents = has_any_compact(question, ["เตรียม", "เอกสาร", "สมัคร"])
        has_application_docs = has_any_compact(evidence["text"], ["เอกสารสมัคร", "เอกสารประกอบ", "แบบฟอร์ม", "เสนอขอรับทุน"])
        if asks_documents and not has_application_docs:
            return (
                f"{UNKNOWN_ANSWER}\n"
                "- ยังไม่พบ “รายการเอกสารที่ต้องเตรียมเพื่อสมัคร” แบบชัดเจนในฐานข้อมูลนี้"
            )
        if not has_any_compact(evidence["text"], ["เอกสาร", "หลักฐาน", "เสนอขอรับทุน", "ขอรับทุน"]):
            return (
                f"{UNKNOWN_ANSWER}\n"
                "- ยังไม่พบรายการเอกสารหรือขั้นตอนสมัครแบบชัดเจนในฐานข้อมูลนี้"
            )
        return f"จากข้อมูลที่พบ สิ่งที่เกี่ยวกับการสมัคร/ยื่นทุนคือ:\n- {snippet}"

    if intent == "amounts":
        amount_chunks = [
            chunk
            for category in ["amounts", "presentation", "publication_rewards", "fund_types"]
            for chunk in by_category.get(category, [])
        ] or chunks
        items = extract_funding_items(amount_chunks)
        if items:
            return "วงเงิน/อัตราทุนที่พบในฐานข้อมูล มีตัวอย่างดังนี้:\n" + "\n".join(items)
        amounts = []
        for chunk in amount_chunks:
            amounts.extend(extract_amounts(chunk["text"]))
        if amounts:
            unique = list(dict.fromkeys(amounts))[:10]
            return "พบตัวเลขวงเงินที่เกี่ยวข้อง เช่น " + ", ".join(unique)
        return UNKNOWN_ANSWER

    if intent == "overview":
        evidence = find_evidence_chunk(chunks, ["เรื่อง", "ประกาศ", "กองทุนวิจัย", "นวัตกรรม", "บริการวิชาการ", "ประเภททุน"], fallback=True)
        snippet = evidence_window(evidence["text"], ["เรื่อง", "ประเภททุน", "กองทุนวิจัย"], 520) if evidence else clean_snippet(chunks[0]["text"], 520)
        return (
            "ประกาศนี้เกี่ยวกับหลักเกณฑ์การใช้จ่ายเงินกองทุนวิจัย นวัตกรรม และบริการวิชาการ "
            "ของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น\n"
            f"- ข้อมูลที่พบ: {snippet}"
        )

    lines = ["จากข้อมูลที่ค้นเจอ มีประเด็นที่เกี่ยวข้องดังนี้"]

    preferred = [
        *(INTENT_CATEGORY_HINTS.get(intent, [])),
        "fund_types",
        "amounts",
        "rules",
        "presentation",
        "publication_rewards",
        "definitions",
        "overview",
        "other",
    ]
    preferred = list(dict.fromkeys(preferred))
    bullet_count = 0
    for category in preferred:
        if category not in by_category:
            continue
        label = CATEGORY_LABELS.get(category, category)
        best = by_category[category][0]
        snippet = clean_snippet(best["text"])
        amounts = extract_amounts(best["text"])
        if amounts:
            snippet = f"{snippet} เช่น {', '.join(amounts[:4])}"
        lines.append(f"- {label}: {snippet}")
        bullet_count += 1
        if bullet_count >= 5:
            break

    if bullet_count == 0:
        for chunk in chunks[:4]:
            lines.append(f"- {clean_snippet(chunk['text'])}")

    return "\n".join(lines)


def extract_pdf_pages(path: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    with pdfplumber.open(str(path)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = " ".join(text.split())
            if text:
                pages.append({"page": index, "text": text})
    return pages


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


THAI_STOPWORDS = (
    "แล้ว", "มี", "อะไร", "ที่", "ฉัน", "เรา", "ต้อง", "รู้", "บ้าง", "ไหม",
    "มั้ย", "คือ", "เกี่ยวกับ", "ประกาศนี้", "ใน", "ของ", "ให้", "หน่อย",
    "ก่อน", "และ", "หรือ", "จาก", "ได้", "การ",
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def expanded_terms(question: str) -> list[str]:
    compact = normalize_text(question)
    terms = []
    for term in re.findall(r"[A-Za-z0-9ก-๙]{2,}", question.lower()):
        clean = term
        for stopword in THAI_STOPWORDS:
            clean = clean.replace(stopword, "")
        if len(clean) >= 2:
            terms.append(clean)

    expansions = {
        "ยื่นทุน": ["ขอรับทุน", "เสนอขอรับทุน", "ทุน", "หลักเกณฑ์", "เงื่อนไข"],
        "ขอทุน": ["ขอรับทุน", "เสนอขอรับทุน", "ทุน", "หลักเกณฑ์", "เงื่อนไข"],
        "สมัครทุน": ["ขอรับทุน", "เสนอขอรับทุน", "ทุน", "หลักเกณฑ์", "เงื่อนไข"],
        "ต้องรู้อะไร": ["หลักเกณฑ์", "เงื่อนไข", "ประเภททุน", "อัตราการให้เงินทุน"],
        "ก่อนยื่น": ["หลักเกณฑ์", "เงื่อนไข", "ขอรับทุน", "เสนอขอรับทุน"],
        "ประเภททุน": ["ประเภททุน", "อัตราการให้เงินทุน", "จำนวนทุน", "วงเงินรวม"],
        "หน่วยงาน": ["มหาวิทยาลัยขอนแก่น", "วิทยาลัยการคอมพิวเตอร์", "ประกาศมหาวิทยาลัยขอนแก่น"],
        "ของใคร": ["มหาวิทยาลัยขอนแก่น", "วิทยาลัยการคอมพิวเตอร์", "ประกาศมหาวิทยาลัยขอนแก่น"],
    }
    for trigger, values in expansions.items():
        if trigger in compact:
            terms.extend(values)

    deduped = []
    for term in terms:
        if term and term not in deduped:
            deduped.append(term)
    return deduped


def keyword_score(question: str, text: str) -> float:
    terms = expanded_terms(question)
    if not terms:
        return 0.0
    compact_text = normalize_text(text)
    hits = 0.0
    for term in terms:
        compact_term = normalize_text(term)
        if compact_term in compact_text:
            hits += 1.0 if len(compact_term) < 5 else 1.35
    return min(1.0, hits / max(2.0, len(terms) * 0.55))


def rank_chunks(
    question: str,
    rows: list[sqlite3.Row],
    selected_category: str | None = None,
) -> list[dict[str, Any]]:
    question_embedding = ollama_embed(question)
    intent = detect_question_intent(question)
    hinted_categories = INTENT_CATEGORY_HINTS.get(intent, [])
    ranked = []
    for row in rows:
        chunk = dict(row)
        semantic = cosine_similarity(question_embedding, json.loads(row["embedding"]))
        lexical = keyword_score(question, row["text"])
        category_boost = 0.0
        if selected_category and row["category"] == selected_category:
            category_boost += 0.20
        elif not selected_category and row["category"] in hinted_categories:
            category_boost += max(0.03, 0.12 - (hinted_categories.index(row["category"]) * 0.025))
        chunk["semantic_score"] = round(semantic, 4)
        chunk["keyword_score"] = round(lexical, 4)
        chunk["category_boost"] = round(category_boost, 4)
        chunk["score"] = round((semantic * 0.66) + (lexical * 0.26) + category_boost, 4)
        ranked.append(chunk)
    ranked.sort(key=lambda chunk: chunk["score"], reverse=True)
    return ranked


def debug_summary(question: str, chunks: list[dict[str, Any]], selected_category: str | None) -> dict[str, Any]:
    intent = detect_question_intent(question)
    return {
        "intent": intent,
        "selected_category": selected_category,
        "expanded_terms": expanded_terms(question),
        "top_chunks": [
            {
                "title": chunk["title"],
                "page": chunk["page"],
                "category": chunk["category"],
                "subcategory": chunk["subcategory"],
                "score": chunk.get("score"),
                "semantic_score": chunk.get("semantic_score"),
                "keyword_score": chunk.get("keyword_score"),
                "category_boost": chunk.get("category_boost"),
            }
            for chunk in chunks[:6]
        ],
    }


def select_sources_for_answer(intent: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    limits = {
        "agency": 2,
        "source": 5,
        "overview": 3,
        "eligibility": 3,
        "application_requirements": 3,
        "amounts": 5,
        "presentation": 4,
        "publication_rewards": 4,
    }
    limit = limits.get(intent, 4)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for chunk in chunks:
        if intent == "agency" and not has_any_compact(chunk["text"], ["มหาวิทยาลัยขอนแก่น", "วิทยาลัยการคอมพิวเตอร์", "ประกาศมหาวิทยาลัยขอนแก่น"]):
            continue
        key = (chunk["document_id"], chunk["page"], chunk["category"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "page": chunk["page"],
                "category": chunk["category"],
                "subcategory": chunk["subcategory"],
                "level": chunk["level"],
                "score": chunk["score"],
            }
        )
        if len(selected) >= limit:
            break
    return selected


def save_qa_log(
    question: str,
    answer: str,
    mode: str,
    category: str | None,
    intent: str,
    elapsed_ms: int,
    sources: list[dict[str, Any]],
    debug: dict[str, Any],
) -> int:
    with db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO qa_logs
            (question, answer, mode, category, intent, elapsed_ms, sources_json, debug_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                question,
                answer,
                mode,
                category,
                intent,
                elapsed_ms,
                json.dumps(sources, ensure_ascii=False),
                json.dumps(debug, ensure_ascii=False),
                int(time.time()),
            ),
        )
        return int(cursor.lastrowid)


def save_document_chunks(
    document_id: str,
    title: str,
    source_type: str,
    source_ref: str | None,
    page_chunks: list[dict[str, Any]],
    forced_category: str | None = None,
) -> dict[str, Any]:
    if not page_chunks:
        raise HTTPException(status_code=400, detail="No text to index")

    rows = []
    for item in page_chunks:
        meta = classify_chunk(item["text"], item["page"], forced_category)
        rows.append(
            {
                "page": item["page"],
                "text": item["text"],
                "category": meta["category"],
                "subcategory": meta["subcategory"],
                "level": meta["level"],
                "tags": json.dumps(meta["tags"], ensure_ascii=False),
                "embedding": json.dumps(ollama_embed(item["text"])),
            }
        )

    pages_with_text = len({row["page"] for row in rows})
    category_counts: dict[str, int] = {}
    for row in rows:
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
    document_category = max(category_counts, key=category_counts.get)
    with db() as conn:
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.execute(
            """
            INSERT INTO documents
            (id, title, source_type, source_ref, category, pages_with_text, chunks, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                title,
                source_type,
                source_ref,
                document_category,
                pages_with_text,
                len(rows),
                int(time.time()),
            ),
        )
        conn.executemany(
            """
            INSERT INTO chunks
            (document_id, title, page, category, subcategory, level, tags, text, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    document_id,
                    title,
                    row["page"],
                    row["category"],
                    row["subcategory"],
                    row["level"],
                    row["tags"],
                    row["text"],
                    row["embedding"],
                )
                for row in rows
            ],
        )

    return {
        "id": document_id,
        "title": title,
        "source_type": source_type,
        "source_ref": source_ref,
        "category": document_category,
        "pages_with_text": pages_with_text,
        "chunks": len(rows),
    }


@app.get("/", include_in_schema=False)
def web_client() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "ollama_url": OLLAMA_URL,
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "database": str(DB_PATH),
    }


@app.get("/documents", dependencies=[Depends(require_api_key)])
def list_documents() -> dict[str, Any]:
    init_db()
    with db() as conn:
        rows = conn.execute(
            """
            SELECT id, title, source_type, source_ref, category, pages_with_text, chunks, created_at
            FROM documents
            ORDER BY created_at DESC
            """
        ).fetchall()
    return {"documents": [dict(row) for row in rows]}


@app.get("/categories", dependencies=[Depends(require_api_key)])
def list_categories() -> dict[str, Any]:
    init_db()
    with db() as conn:
        rows = conn.execute(
            """
            SELECT category, COUNT(*) AS chunks, MIN(level) AS best_level
            FROM chunks
            GROUP BY category
            ORDER BY chunks DESC
            """
        ).fetchall()
    existing = {
        row["category"]: {
            "id": row["category"],
            "label": CATEGORY_LABELS.get(row["category"], row["category"]),
            "chunks": row["chunks"],
            "level": row["best_level"],
        }
        for row in rows
    }
    return {
        "categories": [
            existing.get(
                category_id,
                {"id": category_id, "label": label, "chunks": 0, "level": 3},
            )
            for category_id, label in CATEGORY_LABELS.items()
        ]
    }


@app.get("/debug/search", dependencies=[Depends(require_api_key)])
def debug_search(q: str, top_k: int = 8, category: str | None = None) -> dict[str, Any]:
    init_db()
    with db() as conn:
        rows = conn.execute(
            """
            SELECT document_id, title, page, category, subcategory, level, tags, text, embedding
            FROM chunks
            """
        ).fetchall()
    ranked = rank_chunks(q, rows, category)[: max(1, min(top_k, 12))]
    return {
        "query": q,
        "terms": expanded_terms(q),
        "results": [
            {
                "title": row["title"],
                "page": row["page"],
                "category": row["category"],
                "subcategory": row["subcategory"],
                "level": row["level"],
                "score": row["score"],
                "semantic_score": row["semantic_score"],
                "keyword_score": row["keyword_score"],
                "snippet": row["text"][:420],
            }
            for row in ranked
        ],
    }


@app.post("/knowledge/text", dependencies=[Depends(require_api_key)])
def add_knowledge_text(request: KnowledgeTextRequest) -> dict[str, Any]:
    text = " ".join(request.text.split())
    title = request.title.strip()
    if not title or not text:
        raise HTTPException(status_code=400, detail="Title and text are required")

    doc_hash = f"{title}\n{request.source_ref or ''}\n{text}".encode("utf-8")
    document_id = hashlib.sha256(doc_hash).hexdigest()[:16]
    page_chunks = [
        {"page": request.page, "text": chunk}
        for chunk in chunk_text(text)
    ]
    return save_document_chunks(
        document_id=document_id,
        title=title,
        source_type="database",
        source_ref=request.source_ref,
        page_chunks=page_chunks,
        forced_category=request.category,
    )


@app.post("/documents/upload", dependencies=[Depends(require_api_key)])
def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    # Admin-only helper for the prototype. Public users should call /chat only.
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    raw = file.file.read()
    doc_id = hashlib.sha256(raw).hexdigest()[:16]
    stored_path = DOCUMENT_DIR / f"{doc_id}-{safe_name}"
    with stored_path.open("wb") as output:
        output.write(raw)

    pages = extract_pdf_pages(stored_path)
    if not pages:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="No selectable text found. This PDF probably needs OCR first.",
        )

    page_chunks: list[dict[str, Any]] = []
    for page in pages:
        for text in chunk_text(page["text"]):
            page_chunks.append({"page": page["page"], "text": text})

    result = save_document_chunks(
        document_id=doc_id,
        title=safe_name,
        source_type="pdf",
        source_ref=str(stored_path),
        page_chunks=page_chunks,
    )
    result["stored_path"] = str(stored_path)
    return result


@app.delete("/documents/{document_id}", dependencies=[Depends(require_api_key)])
def delete_document(document_id: str) -> dict[str, Any]:
    init_db()
    with db() as conn:
        document = conn.execute(
            "SELECT * FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if document:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    path = Path(document["source_ref"] or "")
    if document["source_type"] == "pdf" and path.exists() and path.is_file():
        path.unlink()
    return {"deleted": document_id}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(request: ChatRequest) -> ChatResponse:
    started_at = time.perf_counter()
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    init_db()
    with db() as conn:
        rows = conn.execute(
            """
            SELECT document_id, title, page, category, subcategory, level, tags, text, embedding
            FROM chunks
            """
        ).fetchall()
    if not rows:
        raise HTTPException(status_code=400, detail="No knowledge records found")

    selected_category = request.category if request.category in CATEGORY_LABELS else None
    ranked = rank_chunks(question, rows, selected_category)
    limit = max(4, min(request.top_k, 10))
    if selected_category:
        category_ranked = [chunk for chunk in ranked if chunk["category"] == selected_category]
        top_chunks = (category_ranked or ranked)[:limit]
    else:
        top_chunks = ranked[:limit]
    intent = detect_question_intent(question)
    debug = debug_summary(question, top_chunks, selected_category)
    if request.mode == "llm":
        answer = ollama_chat(question, top_chunks)
    else:
        answer = fast_answer(question, top_chunks)
    if answer.startswith(UNKNOWN_ANSWER):
        sources = []
    else:
        sources = select_sources_for_answer(intent, top_chunks)
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    log_id = save_qa_log(
        question=question,
        answer=answer,
        mode=request.mode,
        category=selected_category,
        intent=intent,
        elapsed_ms=elapsed_ms,
        sources=sources,
        debug=debug,
    )
    return ChatResponse(
        answer=answer,
        sources=sources,
        elapsed_ms=elapsed_ms,
        log_id=log_id,
        intent=intent,
        debug=debug,
    )


@app.post("/feedback", dependencies=[Depends(require_api_key)])
def save_feedback(request: FeedbackRequest) -> dict[str, Any]:
    rating = request.rating.strip().lower()
    allowed = {"correct", "wrong", "should_unknown", "unclear"}
    if rating not in allowed:
        raise HTTPException(status_code=400, detail=f"rating must be one of: {', '.join(sorted(allowed))}")
    with db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO feedback
            (log_id, question, answer, rating, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                request.log_id,
                request.question.strip(),
                request.answer.strip(),
                rating,
                (request.note or "").strip() or None,
                int(time.time()),
            ),
        )
    return {"ok": True, "id": int(cursor.lastrowid)}


@app.get("/logs", dependencies=[Depends(require_api_key)])
def list_logs(limit: int = 30) -> dict[str, Any]:
    init_db()
    safe_limit = max(1, min(limit, 100))
    with db() as conn:
        rows = conn.execute(
            """
            SELECT id, question, answer, mode, category, intent, elapsed_ms, sources_json, debug_json, created_at
            FROM qa_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
        feedback_rows = conn.execute(
            """
            SELECT log_id, rating, note, created_at
            FROM feedback
            WHERE log_id IS NOT NULL
            ORDER BY id DESC
            """
        ).fetchall()
    feedback_by_log: dict[int, list[dict[str, Any]]] = {}
    for row in feedback_rows:
        feedback_by_log.setdefault(row["log_id"], []).append(
            {"rating": row["rating"], "note": row["note"], "created_at": row["created_at"]}
        )
    logs = []
    for row in rows:
        logs.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "mode": row["mode"],
                "category": row["category"],
                "intent": row["intent"],
                "elapsed_ms": row["elapsed_ms"],
                "sources": json.loads(row["sources_json"] or "[]"),
                "debug": json.loads(row["debug_json"] or "{}"),
                "feedback": feedback_by_log.get(row["id"], []),
                "created_at": row["created_at"],
            }
        )
    return {"logs": logs}


@app.post("/test/batch", dependencies=[Depends(require_api_key)])
def batch_test(request: BatchTestRequest) -> dict[str, Any]:
    questions = [question.strip() for question in request.questions if question.strip()]
    if not questions:
        raise HTTPException(status_code=400, detail="questions is required")
    if len(questions) > 30:
        raise HTTPException(status_code=400, detail="Run at most 30 questions at a time")
    results = []
    previous_result: dict[str, Any] | None = None
    for question in questions:
        if previous_result and is_source_follow_up(question):
            sources = previous_result.get("sources", [])
            results.append(
                {
                    "question": question,
                    "answer": answer_from_previous_sources(sources),
                    "sources": sources,
                    "elapsed_ms": 0,
                    "log_id": None,
                    "intent": "source",
                    "debug": {
                        "intent": "source",
                        "selected_category": request.category,
                        "expanded_terms": [],
                        "top_chunks": [],
                        "note": "answered from the previous batch result sources",
                    },
                }
            )
            continue

        result = chat(ChatRequest(question=question, mode=request.mode, category=request.category, top_k=10)).model_dump()
        result["question"] = question
        results.append(result)
        previous_result = result
    return {"count": len(results), "results": results}
