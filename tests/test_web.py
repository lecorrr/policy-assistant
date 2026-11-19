import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from policy_assistant.adapters.chat import ModelError
from policy_assistant.adapters.db import DatabaseError
from policy_assistant.answering import Assistant
from policy_assistant.auth import SessionSigner
from policy_assistant.fakes import ExtractiveGenerator, HashingEmbedder, OverlapReranker
from policy_assistant.search import AccessViolation, Retriever
from policy_assistant.web.app import COOKIE, STATIC, Services, create_app


@pytest.fixture
def services(memory_store, lemmatizer, users):
    retriever = Retriever(
        memory_store, HashingEmbedder(lemmatizer, 64), OverlapReranker(lemmatizer), lemmatizer
    )
    assistant = Assistant(retriever, {"fake": ExtractiveGenerator(lemmatizer)}, "fake")
    return Services(assistant, users, SessionSigner("k" * 40, 3600), memory_store)


@pytest.fixture
def client(services):
    return TestClient(create_app(services))


def sign_in(client, login: str) -> None:
    response = client.post("/api/login", json={"login": login, "password": f"demo-{login}"})
    assert response.status_code == 200, response.text


def test_page_and_script_are_served_with_strict_csp(client):
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert 'id="ask-form"' in page.text and "/static/app.js" in page.text
    assert page.headers["content-security-policy"].startswith("default-src 'self'")
    script = client.get("/static/app.js")
    assert script.status_code == 200 and "javascript" in script.headers["content-type"]
    assert "innerHTML" not in script.text  # server text only ever goes in as text


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_script_parses():
    subprocess.run(["node", "--check", str(STATIC / "app.js")], check=True)


def test_asking_needs_session(client):
    assert client.get("/api/me").status_code == 401
    assert client.post("/api/ask", json={"question": "Що?"}).status_code == 401
    client.cookies.set(COOKIE, "olena.9999999999.forged-signature-forged-signature-forged-sig")
    assert client.get("/api/me").status_code == 401


def test_wrong_password_and_unknown_login_get_same_response(client):
    wrong = client.post("/api/login", json={"login": "olena", "password": "nope-nope"})
    unknown = client.post("/api/login", json={"login": "ghost", "password": "nope-nope"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_sign_in_sets_strict_http_only_cookie_and_reports_roles(client):
    response = client.post("/api/login", json={"login": "kateryna", "password": "demo-kateryna"})
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    me = response.json()
    assert me["roles"] == ["risk", "sales", "staff"] and me["models"] == ["fake"]
    assert client.get("/api/me").json()["login"] == "kateryna"


def test_answers_come_with_citations(client):
    sign_in(client, "olena")
    response = client.post(
        "/api/ask",
        json={"question": "На скільки відсоткових пунктів менеджер може знизити ставку?"},
    )
    body = response.json()
    assert response.status_code == 200 and not body["refused"]
    assert body["citations"][0]["label"] == "sales-discounts, розд. 1"
    assert "2 відсоткові пункти" in body["citations"][0]["excerpt"]
    assert body["timings_ms"]["total_ms"] >= 0


def test_sales_user_gets_nothing_from_hr(client, memory_store):
    sign_in(client, "olena")
    body = client.post("/api/ask", json={"question": "Який оклад менеджера з продажу G2?"}).json()
    assert "28 000" not in str(body)
    searched = {roles for roles, _, _ in memory_store.calls}
    assert searched == {("sales", "staff")}


def test_request_cannot_choose_roles(client):
    sign_in(client, "olena")
    response = client.post("/api/ask", json={"question": "Оклади?", "roles": ["hr"]})
    assert response.status_code == 422  # unknown fields are refused, not ignored


def test_documents_list_only_what_user_may_read(client):
    sign_in(client, "andrii")
    assert [d["id"] for d in client.get("/api/documents").json()] == ["staff-handbook"]


def test_logout_ends_session(client):
    sign_in(client, "olena")
    client.post("/api/logout")
    assert client.get("/api/me").status_code == 401


@pytest.mark.parametrize(
    "error, status",
    [
        (ModelError("model down"), 502),
        (DatabaseError("db down"), 503),
        (AccessViolation("row leaked"), 500),
    ],
)
def test_failures_map_to_clear_statuses(client, services, error, status):
    def broken(*args, **kwargs):
        raise error

    services.assistant.ask = broken
    sign_in(client, "olena")
    response = client.post("/api/ask", json={"question": "Що?"})
    assert response.status_code == status
    if status == 500:
        assert "row leaked" not in response.text  # details stay in the log


def test_bad_input(client):
    sign_in(client, "olena")
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "а" * 1001}).status_code == 422
    assert client.post("/api/ask", json={"question": "Що?", "model": "other"}).status_code == 400


def test_health(client, services):
    assert client.get("/api/health").json()["status"] == "ok"

    def down():
        raise DatabaseError("no database")

    services.store.ping = down
    assert client.get("/api/health").status_code == 503
