from collections.abc import Callable

import httpx

from policy_assistant import __version__
from policy_assistant.adapters import db
from policy_assistant.adapters.chat import has_credentials
from policy_assistant.config import Settings
from policy_assistant.corpus import load_corpus
from policy_assistant.evaluation import QUESTIONS_FILE
from policy_assistant.evaluation.cases import load_eval_set
from policy_assistant.wiring import Context


def run_doctor(context: Context) -> int:
    settings = context.settings
    results: list[tuple[str, str, str]] = []

    def check(name: str, run: Callable[[], str]) -> None:
        try:
            results.append(("OK", name, run()))
        except _Warning as warning:
            results.append(("WARN", name, str(warning)))
        except Exception as error:  # every check reports and the next one still runs
            results.append(("FAIL", name, str(error) or type(error).__name__))

    print(f"policy-assistant {__version__}")
    print(
        f"  database {settings.db_host}:{settings.db_port}/{settings.db_name}, admin "
        f"{settings.db_admin_user}, reader {settings.db_reader_user}"
    )
    print(
        f"  embeddings {settings.embedding_model} ({settings.embedding_dims}) via "
        f"{settings.ollama_url}; reranker {settings.reranker}; top {settings.top_k} of "
        f"{settings.rerank_top}"
    )
    secrets_set = [
        name
        for name in (
            "db_admin_password",
            "db_reader_password",
            "session_secret",
            "openai_api_key",
            "anthropic_api_key",
        )
        if getattr(settings, name)
    ]
    print(f"  secrets set: {', '.join(secrets_set) or 'none'}")

    check(
        "config",
        lambda: (
            f"{len(context.models.models)} models, {len(context.users.users)} users, "
            f"roles {', '.join(context.users.roles)}"
        ),
    )
    check("corpus", lambda: _doctor_corpus(context))
    check("database", lambda: _doctor_database(settings))
    check("ollama", lambda: _doctor_ollama(context))
    check("reranker", lambda: _doctor_reranker(settings))
    for name, model in context.models.models.items():
        check(f"model {name}", lambda model=model: _doctor_model(model, context))
    check("evaluation set", lambda: f"{len(load_eval_set(QUESTIONS_FILE).questions)} questions")

    for status, name, detail in results:
        print(f"{status:4} {name}: {detail}")
    return 1 if any(status == "FAIL" for status, _, _ in results) else 0


class _Warning(Exception):
    pass


def _doctor_corpus(context: Context) -> str:
    settings = context.settings
    if not settings.law_file.is_file():
        raise _Warning(f"{settings.law_file} is missing: run `policy-assistant fetch-law`")
    documents = load_corpus(settings.corpus_dir, settings.law_file, set(context.users.roles))
    law = next(d for d in documents if d.kind == "law")
    return f"{len(documents) - 1} policies and the law ({law.version})"


def _doctor_database(settings: Settings) -> str:
    check = db.check_reader(settings.admin_dsn, settings.reader_dsn)
    if check.problems:
        raise RuntimeError("; ".join(check.problems))
    counts = ", ".join(f"{role} {n}" for role, n in check.chunks_by_role.items()) or "no chunks yet"
    if not check.chunks_by_role:
        raise _Warning(
            f"pgvector {check.pgvector_version}, row-level security on; {counts}: run `ingest`"
        )
    return (
        f"pgvector {check.pgvector_version}; row-level security on; the reader sees nothing "
        f"without roles; chunks by role: {counts}"
    )


def _doctor_ollama(context: Context) -> str:
    settings = context.settings
    try:
        response = httpx.get(f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise RuntimeError(f"not reachable at {settings.ollama_url}: {error}") from None
    pulled = {model["name"] for model in response.json().get("models", [])}
    wanted = [settings.embedding_model] + [
        m.model for m in context.models.models.values() if m.provider == "ollama"
    ]
    missing = [name for name in wanted if name not in pulled and f"{name}:latest" not in pulled]
    if missing:
        raise RuntimeError(
            "missing models: " + ", ".join(f"`ollama pull {name}`" for name in missing)
        )
    return f"{', '.join(wanted)} pulled"


def _doctor_reranker(settings: Settings) -> str:
    if settings.reranker == "none":
        raise _Warning("reranking is off (POLICY_RERANKER=none)")
    stem = settings.reranker.split("/")[-1].removesuffix("-int8").lower()
    cached = any(stem in path.name.lower() for path in settings.models_cache.glob("*"))
    if not cached:
        raise _Warning(
            f"{settings.reranker} downloads on first use (about 0.6 GB) into "
            f"{settings.models_cache}"
        )
    return f"{settings.reranker} cached in {settings.models_cache}"


def _doctor_model(model, context: Context) -> str:
    if not has_credentials(model, context.settings):
        key = "OPENAI_API_KEY" if model.provider == "openai" else "ANTHROPIC_API_KEY"
        raise _Warning(f"{model.provider} {model.model}: {key} is not set, so it is skipped")
    price = model.price
    return f"{model.provider} {model.model}, ${price.input}/${price.output} per million tokens"
