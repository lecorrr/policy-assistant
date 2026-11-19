import time
from dataclasses import dataclass

from policy_assistant.config import AnswerModel, Settings
from policy_assistant.contracts import Usage


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class Generation:
    text: str
    usage: Usage
    elapsed_ms: float


def has_credentials(config: AnswerModel, settings: Settings) -> bool:
    if config.provider == "openai":
        return settings.openai_api_key is not None
    if config.provider == "anthropic":
        return settings.anthropic_api_key is not None
    return True


class ChatGenerator:
    def __init__(self, name: str, config: AnswerModel, settings: Settings) -> None:
        self.name = name
        self.config = config
        self._model = _build(name, config, settings)

    def generate(self, system: str, user: str) -> Generation:
        from langchain_core.messages import HumanMessage, SystemMessage

        started = time.perf_counter()
        try:
            message = self._model.invoke([SystemMessage(system), HumanMessage(user)])
        except Exception as error:  # providers raise their own exception types
            raise ModelError(
                f"{self.name} ({self.config.provider} {self.config.model}) failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        usage = message.usage_metadata or {}
        return Generation(
            text=message.text,
            usage=Usage(usage.get("input_tokens"), usage.get("output_tokens")),
            elapsed_ms=elapsed_ms,
        )


def _build(name: str, config: AnswerModel, settings: Settings):
    options = dict(config.options)
    timeout_s = settings.request_timeout_s
    if config.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model,
            base_url=settings.ollama_url,
            client_kwargs={"timeout": timeout_s},
            **options,
        )
    if config.provider == "openai":
        if settings.openai_api_key is None:
            raise ModelError(f"{name} needs OPENAI_API_KEY in .env")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model,
            api_key=settings.openai_api_key,
            timeout=timeout_s,
            max_retries=2,
            use_responses_api=True,  # OpenAI's recommended API for its reasoning models
            **options,
        )
    if config.provider == "anthropic":
        if settings.anthropic_api_key is None:
            raise ModelError(f"{name} needs ANTHROPIC_API_KEY in .env")
        from langchain_anthropic import ChatAnthropic

        # room for adaptive thinking; only generated tokens are billed
        options.setdefault("max_tokens", 2048)
        return ChatAnthropic(
            model=config.model,
            api_key=settings.anthropic_api_key,
            timeout=timeout_s,
            max_retries=2,
            **options,
        )
    raise ModelError(f"{name}: unknown provider {config.provider!r}")
