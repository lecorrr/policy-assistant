from collections.abc import Callable
from dataclasses import dataclass

from policy_assistant.adapters import db
from policy_assistant.adapters.chat import ChatGenerator, ModelError, has_credentials
from policy_assistant.adapters.embeddings import OllamaEmbedder
from policy_assistant.adapters.reranker import build_reranker
from policy_assistant.answering import Assistant
from policy_assistant.config import ConfigError, ModelsConfig, Settings, UsersConfig
from policy_assistant.search import Retriever
from policy_assistant.text import Lemmatizer

FAKE_MODEL = "fake"


@dataclass
class Context:
    settings: Settings
    models: ModelsConfig
    users: UsersConfig


@dataclass
class Built:
    assistant: Assistant | None
    retriever: Retriever
    store: db.ReaderStore
    embedding_model: str
    closers: list[Callable[[], None]]

    def close(self) -> None:
        for close in self.closers:
            close()


def build_services(
    context: Context, *, fake: bool, models: list[str] | None = None, need_generators: bool = True
) -> Built:
    settings = context.settings
    lemmatizer = Lemmatizer()
    embedder = make_embedder(settings, lemmatizer, fake)
    store = db.ReaderStore(settings.reader_dsn)
    closers = [store.close]
    if hasattr(embedder, "close"):
        closers.append(embedder.close)
    stored_model = store.embedding_model()
    if stored_model is None:
        store.close()
        raise db.DatabaseError("the corpus is not stored yet: run `policy-assistant ingest`")
    if stored_model != embedder.model:
        store.close()
        raise db.DatabaseError(
            f"the corpus was embedded with {stored_model} but questions would be embedded with "
            f"{embedder.model}: run `policy-assistant ingest`{' --fake' if fake else ''} again"
        )
    if fake:
        from policy_assistant.fakes import OverlapReranker

        reranker = OverlapReranker(lemmatizer)
    else:
        reranker = build_reranker(
            settings.reranker, settings.models_cache, settings.reranker_threads
        )
    retriever = Retriever(
        store,
        embedder,
        reranker,
        lemmatizer,
        candidates=settings.candidates,
        rerank_top=settings.rerank_top,
        top_k=settings.top_k,
    )
    assistant = None
    if need_generators:
        generators = build_generators(context, lemmatizer, fake=fake, only=models)
        default = (
            FAKE_MODEL if fake else (models[0] if models else default_model(context, generators))
        )
        assistant = Assistant(retriever, generators, default)
    return Built(assistant, retriever, store, embedder.model, closers)


def make_embedder(settings: Settings, lemmatizer: Lemmatizer, fake: bool):
    if fake:
        from policy_assistant.fakes import HashingEmbedder

        return HashingEmbedder(lemmatizer, settings.embedding_dims)
    return OllamaEmbedder(
        settings.ollama_url,
        settings.embedding_model,
        settings.embedding_dims,
        timeout_s=settings.request_timeout_s,
    )


def build_generators(
    context: Context, lemmatizer: Lemmatizer, *, fake: bool, only: list[str] | None
) -> dict:
    if fake:
        from policy_assistant.fakes import ExtractiveGenerator

        return {FAKE_MODEL: ExtractiveGenerator(lemmatizer, FAKE_MODEL)}
    generators = {}
    for name, model in context.models.models.items():
        if only is not None and name not in only:
            continue
        if has_credentials(model, context.settings):
            generators[name] = ChatGenerator(name, model, context.settings)
    if only:
        missing = [name for name in only if name not in generators]
        if missing:
            raise ModelError(f"cannot run {', '.join(missing)}: unknown, or its API key is not set")
    if not generators:
        raise ModelError("no answer model can run: check config/models.yaml and the API keys")
    return generators


def default_model(context: Context, generators: dict) -> str:
    return context.models.default if context.models.default in generators else sorted(generators)[0]


def check_models_runnable(names: list[str], context: Context, *, explicit: bool) -> None:
    for name in list(names):
        model = context.models.models.get(name)
        if model is None:
            raise ConfigError(f"no model {name!r} in {context.settings.models_file}")
        if not has_credentials(model, context.settings):
            if explicit:
                raise ConfigError(f"{name} needs its API key in .env")
            print(f"Skipping {name}: its API key is not set.")
            names.remove(name)
    if not names:
        raise ConfigError("no model to evaluate: set an API key or run the local model")


def build_judge(choice: str | None, context: Context):
    name = context.models.judge if choice is None else choice
    if not name or name == "none":
        return None
    model = context.models.models.get(name)
    if model is None:
        raise ConfigError(f"no judge model {name!r} in {context.settings.models_file}")
    if not has_credentials(model, context.settings):
        print(f"Judging skipped: {name} needs its API key. Faithfulness is reported as not judged.")
        return None
    return ChatGenerator(name, model, context.settings)
