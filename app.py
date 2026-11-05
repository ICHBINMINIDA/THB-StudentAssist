"""
THB StudentAssist web app.

Run:  uvicorn app:app --reload
"""

import json
import logging
import os
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import config
from chatbot import LLMUnavailableError, answer_question, clear_history, stream_answer

logger = logging.getLogger("app")

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="THB StudentAssist")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
# Serve the source PDFs so that citations can link to the exact page.
os.makedirs(config.PDF_DIR, exist_ok=True)
app.mount("/pdf", StaticFiles(directory=config.PDF_DIR), name="pdf")

templates = Jinja2Templates(directory="templates")

SESSION_COOKIE = "session_id"
LLM_DOWN_MESSAGE = "Der Sprachdienst ist gerade nicht erreichbar. Bitte versuche es in Kürze erneut."


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=config.MAX_QUESTION_LENGTH)


async def parse_question(request: Request) -> tuple[str | None, JSONResponse | None]:
    try:
        body = await request.json()
        question = AskRequest(question=str(body.get("question", "")).strip()).question
        return question, None
    except ValidationError:
        return None, JSONResponse(
            {"error": f"Bitte gib eine Frage mit höchstens {config.MAX_QUESTION_LENGTH} Zeichen ein."},
            status_code=422,
        )
    except (json.JSONDecodeError, AttributeError):
        return None, JSONResponse({"error": "Invalid request body."}, status_code=400)


def get_session(request: Request) -> tuple[str, bool]:
    """Return (session_id, is_new). Ignores malformed cookies."""
    sid = request.cookies.get(SESSION_COOKIE, "")
    try:
        uuid.UUID(sid)
        return sid, False
    except ValueError:
        return str(uuid.uuid4()), True


def attach_session(response, session_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        max_age=config.SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=config.COOKIE_SECURE,
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"max_length": config.MAX_QUESTION_LENGTH})


@app.post("/ask")
@limiter.limit(config.RATE_LIMIT)
async def ask(request: Request):
    """Non-streaming endpoint: returns {"answer", "sources"}."""
    question, error = await parse_question(request)
    if error:
        return error
    session_id, is_new = get_session(request)

    try:
        # The LLM call is blocking; run it in a worker thread so the server stays responsive.
        result = await run_in_threadpool(answer_question, question, session_id)
    except LLMUnavailableError:
        return JSONResponse({"error": LLM_DOWN_MESSAGE}, status_code=503)

    response = JSONResponse(result)
    if is_new:
        attach_session(response, session_id)
    return response


@app.post("/ask/stream")
@limiter.limit(config.RATE_LIMIT)
async def ask_stream(request: Request):
    """
    Streaming endpoint. Response is NDJSON, one event per line:
      {"type":"token","text":"..."} ... {"type":"done","sources":[...]}
    or {"type":"error","message":"..."}.
    """
    question, error = await parse_question(request)
    if error:
        return error
    session_id, is_new = get_session(request)

    async def events():
        try:
            async for event in stream_answer(question, session_id):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except LLMUnavailableError:
            yield json.dumps({"type": "error", "message": LLM_DOWN_MESSAGE}, ensure_ascii=False) + "\n"
        except Exception:
            logger.exception("Unexpected error while streaming")
            yield json.dumps({"type": "error", "message": "Interner Fehler."}) + "\n"

    response = StreamingResponse(events(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    if is_new:
        attach_session(response, session_id)
    return response


@app.post("/clear")
@limiter.limit(config.RATE_LIMIT)
async def clear(request: Request):
    session_id, is_new = get_session(request)
    if not is_new:
        clear_history(session_id)
    return JSONResponse({"status": "ok"})


@app.get("/health")
async def health():
    """Liveness + whether the Ollama backend is reachable."""
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
            llm_ok = r.status_code == 200
    except httpx.HTTPError:
        llm_ok = False
    return JSONResponse({"status": "ok" if llm_ok else "degraded", "llm": llm_ok},
                        status_code=200 if llm_ok else 503)
