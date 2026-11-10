"""The adapters against local stand-ins: the Rada API, Ollama, the reranker and chat models."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import SAMPLE_LAW, make_settings
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from policy_assistant.adapters import rada
from policy_assistant.adapters.chat import ChatGenerator, ModelError, has_credentials
from policy_assistant.adapters.db import roles_literal, vector_literal
from policy_assistant.adapters.embeddings import EmbeddingError, OllamaEmbedder
from policy_assistant.adapters.reranker import NoReranker, RerankerError, build_reranker
from policy_assistant.config import AnswerModel
from policy_assistant.corpus import load_law


class StandIn:
    """A local HTTP server whose reply each test sets; it records the requests it gets."""

    def __init__(self) -> None:
        self.reply = (200, {"Content-Type": "text/plain"}, b"")
        self.requests: list[dict] = []
        stand_in = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self._answer()

            def do_POST(self):
                self._answer()

            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                stand_in.requests.append(
                    {"path": self.path, "headers": dict(self.headers), "body": body}
                )
                status, headers, payload = (
                    stand_in.reply(body) if callable(stand_in.reply) else stand_in.reply
                )
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def stand_in():
    server = StandIn()
    yield server
    server.close()


# --- the Rada's open-data API ---


def test_law_download_sends_the_open_data_user_agent(stand_in, tmp_path):
    stand_in.reply = (200, {"Content-Type": "text/plain; charset=utf-8"}, SAMPLE_LAW.encode())
    text = rada.fetch_law_text("1734-19", url=f"{stand_in.url}/laws/show/1734-19.txt")
    assert text == SAMPLE_LAW
    headers = stand_in.requests[0]["headers"]
    assert headers["User-Agent"] == "OpenData"  # exactly; anything else is refused
    assert headers["X-Client"].startswith("policy-assistant/")
    path = tmp_path / "sources" / "1734-19.txt"
    note = rada.save_law_text(text, path, url="u")
    assert path.read_text(encoding="utf-8") == SAMPLE_LAW and len(note["sha256"]) == 64
    assert load_law(path).version.startswith("чинна редакція станом на")


def test_law_download_decodes_windows_1251(stand_in):
    stand_in.reply = (200, {"Content-Type": "text/plain"}, SAMPLE_LAW.encode("cp1251"))
    assert rada.fetch_law_text("x", url=stand_in.url) == SAMPLE_LAW


@pytest.mark.parametrize(
    "reply, message",
    [
        ((302, {"Location": "https://zakon.rada.gov.ua/"}, b""), "refused"),
        ((403, {"Content-Type": "text/html"}, b"no"), "refused"),
        ((500, {"Content-Type": "text/plain"}, b""), "HTTP 500"),
        ((200, {"Content-Type": "text/html"}, b"<html>Captcha</html>"), "not plain text"),
        ((200, {"Content-Type": "text/plain"}, "Щось інше".encode()), "did not return the text"),
    ],
)
def test_law_download_failures_say_what_to_do(stand_in, reply, message):
    stand_in.reply = reply
    with pytest.raises(rada.LawDownloadError, match=message) as error:
        rada.fetch_law_text("x", url=stand_in.url)
    assert "README" in str(error.value)
    assert len(stand_in.requests) == 1  # a redirect is never followed


def test_law_download_without_a_server():
    with pytest.raises(rada.LawDownloadError, match="cannot reach"):
        rada.fetch_law_text("x", url="http://127.0.0.1:9/laws.txt", timeout_s=2)


# --- Ollama embeddings ---


def embed_reply(dims):
    def reply(body):
        texts = json.loads(body)["input"]
        payload = {"embeddings": [[0.1] * dims for _ in texts]}
        return 200, {"Content-Type": "application/json"}, json.dumps(payload).encode()

    return reply


def test_embeddings_in_batches(stand_in):
    stand_in.reply = embed_reply(8)
    embedder = OllamaEmbedder(stand_in.url, "bge-m3", 8, batch_size=2)
    vectors = embedder.embed(["а", "б", "в"])
    assert len(vectors) == 3 and len(vectors[0]) == 8
    assert [json.loads(r["body"])["input"] for r in stand_in.requests] == [["а", "б"], ["в"]]
    assert json.loads(stand_in.requests[0]["body"])["model"] == "bge-m3"


def test_embedding_failures(stand_in):
    stand_in.reply = embed_reply(4)
    with pytest.raises(EmbeddingError, match="4-dimension embeddings"):
        OllamaEmbedder(stand_in.url, "bge-m3", 8).embed(["а"])
    stand_in.reply = (404, {"Content-Type": "application/json"}, b'{"error":"model not found"}')
    with pytest.raises(EmbeddingError, match="ollama pull bge-m3"):
        OllamaEmbedder(stand_in.url, "bge-m3", 8).embed(["а"])
    with pytest.raises(EmbeddingError, match="not reachable"):
        OllamaEmbedder("http://127.0.0.1:9", "bge-m3", 8, timeout_s=2).embed(["а"])


# --- reranker ---


def test_no_reranker_keeps_the_order(tmp_path):
    reranker = build_reranker("none", tmp_path)
    assert isinstance(reranker, NoReranker)
    scores = reranker.scores("q", ["a", "b", "c"])
    assert sorted(range(3), key=lambda i: -scores[i]) == [0, 1, 2]


def test_an_unknown_reranker_model_is_a_clear_error(tmp_path):
    with pytest.raises(RerankerError, match="not-a-model"):
        build_reranker("acme/not-a-model", tmp_path / "models")


# --- chat models ---


def answer_model(provider: str, **options) -> AnswerModel:
    return AnswerModel(
        provider=provider,
        model="m",
        options=options,
        price={"input": 0, "output": 0},
        price_source="test",
    )


def test_cloud_models_need_their_keys():
    settings = make_settings()
    assert has_credentials(answer_model("ollama"), settings)
    assert not has_credentials(answer_model("openai"), settings)
    with pytest.raises(ModelError, match="OPENAI_API_KEY"):
        ChatGenerator("gpt", answer_model("openai"), settings)
    with pytest.raises(ModelError, match="ANTHROPIC_API_KEY"):
        ChatGenerator("claude", answer_model("anthropic"), settings)


def test_each_provider_builds_offline_with_its_options():
    settings = make_settings(openai_api_key="sk-test", anthropic_api_key="sk-ant-test")
    local = ChatGenerator("local", answer_model("ollama", reasoning=False, temperature=0), settings)
    assert local._model.reasoning is False
    gpt = ChatGenerator("gpt", answer_model("openai", reasoning_effort="low"), settings)
    assert gpt._model.reasoning_effort == "low" and gpt._model.use_responses_api
    claude = ChatGenerator("claude", answer_model("anthropic", effort="low"), settings)
    assert claude._model.reasoning_effort == "low" and claude._model.max_tokens == 2048


def test_generation_reads_text_blocks_and_usage():
    generator = ChatGenerator("local", answer_model("ollama"), make_settings())
    reply = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "думаю"},
            {"type": "text", "text": "Відповідь [1]."},
        ],
        usage_metadata={"input_tokens": 3100, "output_tokens": 42, "total_tokens": 3142},
    )
    generator._model = FakeMessagesListChatModel(responses=[reply])
    result = generator.generate("system", "user")
    assert result.text == "Відповідь [1]."
    assert (result.usage.input_tokens, result.usage.output_tokens) == (3100, 42)
    assert result.elapsed_ms >= 0


def test_a_failing_model_names_itself():
    generator = ChatGenerator("local", answer_model("ollama"), make_settings())

    class Broken:
        def invoke(self, messages):
            raise ConnectionError("refused")

    generator._model = Broken()
    with pytest.raises(ModelError, match=r"local .*ConnectionError: refused"):
        generator.generate("s", "u")


# --- database literals ---


def test_role_and_vector_literals_reject_anything_unusual():
    assert roles_literal(["staff", "sales", "staff"]) == "{sales,staff}"
    assert roles_literal([]) == "{}"
    for bad in (["hr}", "x"], ["Sales"], ["a,b"], ["'"]):
        with pytest.raises(ValueError):
            roles_literal(bad)
    assert vector_literal([1, 0.5]) == "[1.0,0.5]"
    with pytest.raises(ValueError, match="NaN"):
        vector_literal([float("nan")])
