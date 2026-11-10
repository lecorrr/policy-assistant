"""The HTTP API and the static page.

Who the user is comes only from the signed session cookie; their roles come from
config/users.yaml on every request. Nothing in a request body can widen what search returns.
"""

import logging
import secrets
from dataclasses import dataclass
from importlib.resources import files
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from policy_assistant import __version__
from policy_assistant.adapters.chat import ModelError
from policy_assistant.adapters.db import DatabaseError
from policy_assistant.adapters.embeddings import EmbeddingError
from policy_assistant.adapters.reranker import RerankerError
from policy_assistant.answering import MAX_QUESTION_CHARS, Assistant, BadQuestion, UnknownModel
from policy_assistant.auth import SessionSigner, hash_password, verify_password
from policy_assistant.config import UsersConfig
from policy_assistant.contracts import Answer, User
from policy_assistant.search import AccessViolation

COOKIE = "policy_session"
log = logging.getLogger("policy_assistant.web")
STATIC = files("policy_assistant.web") / "static"


class DocumentLister(Protocol):
    def documents(self, roles) -> list[dict]: ...

    def ping(self) -> None: ...


@dataclass
class Services:
    assistant: Assistant
    users: UsersConfig
    signer: SessionSigner
    store: DocumentLister
    cookie_secure: bool = False


class LoginIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    login: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class AskIn(BaseModel):
    # Unknown fields are refused, so a client cannot believe it chose its roles.
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    model: str | None = Field(None, max_length=64)


def create_app(services: Services) -> FastAPI:
    # Checked when the login does not exist, so the reply time does not reveal which logins do.
    dummy_hash = hash_password(secrets.token_urlsafe(16))
    app = FastAPI(title="Policy assistant", version=__version__, docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def current_user(request: Request) -> User:
        login = services.signer.login_of(request.cookies.get(COOKIE, ""))
        user = services.users.user(login) if login else None
        if user is None:
            raise HTTPException(401, "Увійдіть, щоб ставити питання.")
        return user

    SignedIn = Annotated[User, Depends(current_user)]

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(str(STATIC / "index.html"), media_type="text/html; charset=utf-8")

    @app.get("/api/health")
    def health() -> dict:
        try:
            services.store.ping()
        except DatabaseError as error:
            raise HTTPException(503, str(error)) from None
        return {"status": "ok", "version": __version__}

    @app.post("/api/login")
    def login(body: LoginIn, response: Response) -> dict:
        entry = services.users.users.get(body.login)
        stored = entry.password_hash if entry else dummy_hash
        if not verify_password(body.password, stored) or entry is None:
            raise HTTPException(401, "Невірний логін або пароль.")
        response.set_cookie(
            COOKIE,
            services.signer.issue(body.login),
            max_age=int(services.signer.max_age_s),
            httponly=True,
            samesite="strict",
            secure=services.cookie_secure,
        )
        return _me(services, services.users.user(body.login))

    @app.post("/api/logout")
    def logout(response: Response) -> dict:
        response.delete_cookie(COOKIE)
        return {"ok": True}

    @app.get("/api/me")
    def me(user: SignedIn) -> dict:
        return _me(services, user)

    @app.get("/api/documents")
    def documents(user: SignedIn) -> list[dict]:
        try:
            return services.store.documents(user.roles)
        except DatabaseError as error:
            raise HTTPException(503, str(error)) from None

    @app.post("/api/ask")
    def ask(body: AskIn, user: SignedIn) -> dict:
        try:
            answer = services.assistant.ask(user, body.question, model=body.model)
        except (BadQuestion, UnknownModel) as error:
            raise HTTPException(400, str(error)) from None
        except ModelError as error:
            log.warning("model failed: %s", error)
            raise HTTPException(502, f"Модель не відповіла: {error}") from None
        except (DatabaseError, EmbeddingError, RerankerError) as error:
            log.warning("search failed: %s", error)
            raise HTTPException(503, f"Пошук недоступний: {error}") from None
        except AccessViolation:
            log.exception("row-level security did not filter a row")
            raise HTTPException(
                500, "Помилка налаштування доступу. Повідомте адміністратора."
            ) from None
        log.info(
            "ask user=%s model=%s refused=%s cited=%s total_ms=%s",
            user.login,
            answer.model,
            answer.refused,
            len(answer.citations),
            answer.timings_ms.get("total_ms"),
        )
        return answer_json(answer)

    return app


def answer_json(answer: Answer) -> dict:
    return {
        "answer": answer.text,
        "refused": answer.refused,
        "model": answer.model,
        "citations": [
            {
                "number": citation.number,
                "label": citation.hit.label,
                "heading": citation.hit.heading,
                "excerpt": citation.hit.body,
                "url": citation.hit.url,
            }
            for citation in answer.citations
        ],
        "timings_ms": answer.timings_ms,
        "usage": {
            "input_tokens": answer.usage.input_tokens,
            "output_tokens": answer.usage.output_tokens,
        },
    }


def _me(services: Services, user: User) -> dict:
    return {
        "login": user.login,
        "name": user.name,
        "title": user.title,
        "roles": list(user.roles),
        "models": sorted(services.assistant.generators),
        "default_model": services.assistant.default_model,
    }
