"""Settings and configuration files, all validated when loaded.

Passwords, keys and addresses come from the environment or .env (prefix POLICY_, plus the
providers' usual OPENAI_API_KEY and ANTHROPIC_API_KEY). The answer models and their prices live in
config/models.yaml, the roles and demo users in config/users.yaml. A misspelt key in either file,
or a POLICY_ variable nobody reads, is an error rather than a silent default.
"""

import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from psycopg.conninfo import make_conninfo
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from policy_assistant.contracts import STAFF_ROLE, User

PROJECT_ROOT = Path(os.environ.get("POLICY_ASSISTANT_HOME") or Path(__file__).resolve().parents[2])
ENV_FILE = PROJECT_ROOT / ".env"

_NAME = r"^[a-z0-9][a-z0-9._-]*$"


class ConfigError(ValueError):
    """Configuration that cannot work; the message names the file or variable."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="POLICY_", env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    db_host: str = "127.0.0.1"
    db_port: int = Field(5442, ge=1, le=65535)
    db_name: str = "policy"
    # The admin owns the tables: it creates the schema and writes the corpus.
    db_admin_user: str = "policy_admin"
    db_admin_password: SecretStr | None = None
    # The app reads as this role, which row-level security applies to.
    db_reader_user: str = "policy_reader"
    db_reader_password: SecretStr | None = None

    ollama_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "bge-m3"
    embedding_dims: int = Field(1024, ge=8, le=4096)
    reranker: str = "BAAI/bge-reranker-v2-m3-int8"  # "none" turns reranking off
    reranker_threads: int | None = Field(None, ge=1, le=64)

    candidates: int = Field(30, ge=5, le=200)  # rows each search method returns
    rerank_top: int = Field(20, ge=1, le=100)  # fused candidates the reranker scores
    top_k: int = Field(5, ge=1, le=20)  # excerpts the model gets
    chunk_chars: int = Field(1200, ge=200, le=4000)

    session_secret: SecretStr | None = None
    session_hours: float = Field(8, gt=0, le=24 * 30)
    cookie_secure: bool = False  # set true when the page is served over HTTPS
    request_timeout_s: float = Field(120, gt=0)

    data_dir: Path = PROJECT_ROOT / "data"
    corpus_dir: Path = PROJECT_ROOT / "corpus" / "policies"
    models_file: Path = PROJECT_ROOT / "config" / "models.yaml"
    users_file: Path = PROJECT_ROOT / "config" / "users.yaml"

    openai_api_key: SecretStr | None = Field(
        None, validation_alias=AliasChoices("OPENAI_API_KEY", "POLICY_OPENAI_API_KEY")
    )
    anthropic_api_key: SecretStr | None = Field(
        None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "POLICY_ANTHROPIC_API_KEY")
    )

    @field_validator(
        "db_admin_password",
        "db_reader_password",
        "session_secret",
        "openai_api_key",
        "anthropic_api_key",
        mode="before",
    )
    @classmethod
    def _empty_is_unset(cls, value: object) -> object:
        # `OPENAI_API_KEY=` in .env means "no key", not an empty key.
        return None if value == "" else value

    @model_validator(mode="after")
    def _check_sizes(self) -> "Settings":
        if self.top_k > self.rerank_top:
            raise ValueError(f"top_k ({self.top_k}) is larger than rerank_top ({self.rerank_top})")
        return self

    @property
    def admin_dsn(self) -> str:
        return self._dsn(self.db_admin_user, self.db_admin_password)

    @property
    def reader_dsn(self) -> str:
        return self._dsn(self.db_reader_user, self.db_reader_password)

    @property
    def law_file(self) -> Path:
        return self.data_dir / "sources" / "1734-19.txt"

    @property
    def models_cache(self) -> Path:
        return self.data_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "evals"

    def _dsn(self, user: str, password: SecretStr | None) -> str:
        secret = password.get_secret_value() if password else None
        return make_conninfo(
            host=self.db_host, port=self.db_port, dbname=self.db_name, user=user, password=secret
        )


# --- config/models.yaml ---

_OPTIONS = {
    "ollama": {"temperature", "reasoning", "num_ctx", "num_predict", "keep_alive"},
    "openai": {"reasoning_effort", "max_tokens", "temperature"},
    "anthropic": {"effort", "max_tokens", "temperature"},
}


class Price(BaseModel):
    """US dollars per million tokens."""

    model_config = ConfigDict(extra="forbid")

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    # Prompts longer than this many tokens pay the long_* prices instead.
    long_prompt_tokens: int | None = Field(None, gt=0)
    long_input: float | None = Field(None, ge=0)
    long_output: float | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _long_prices_together(self) -> "Price":
        given = [self.long_prompt_tokens, self.long_input, self.long_output]
        if any(v is not None for v in given) and any(v is None for v in given):
            raise ValueError("long_prompt_tokens, long_input and long_output go together")
        return self

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        input_price, output_price = self.input, self.output
        if self.long_prompt_tokens is not None and input_tokens > self.long_prompt_tokens:
            input_price, output_price = self.long_input, self.long_output
        return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


class AnswerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama", "openai", "anthropic"]
    model: str = Field(min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)
    price: Price
    price_source: str = Field(min_length=3)

    @model_validator(mode="after")
    def _known_options(self) -> "AnswerModel":
        unknown = sorted(set(self.options) - _OPTIONS[self.provider])
        if unknown:
            allowed = ", ".join(sorted(_OPTIONS[self.provider]))
            raise ValueError(f"unknown options {', '.join(unknown)} for {self.provider}: {allowed}")
        return self


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prices_checked: date
    default: str
    memo: list[str] = Field(min_length=1)  # the models the memo compares, in this order
    judge: str | None = None  # checks faithfulness in eval-answers; null skips the check
    models: dict[str, AnswerModel] = Field(min_length=1)

    @model_validator(mode="after")
    def _references(self) -> "ModelsConfig":
        for name in self.models:
            if not re.match(_NAME, name):
                raise ValueError(f"model name {name!r} must be lower-case letters, digits, . or -")
        named = [("default", self.default), *(("memo", n) for n in self.memo)]
        if self.judge:
            named.append(("judge", self.judge))
        for field_name, name in named:
            if name not in self.models:
                raise ValueError(f"{field_name}: {name!r} is not one of the models below")
        return self


# --- config/users.yaml ---


class UserEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    title: str = ""
    roles: list[str] = Field(default_factory=list)
    password_hash: str = Field(pattern=r"^scrypt\$")


class UsersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roles: list[str] = Field(min_length=1)
    users: dict[str, UserEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> "UsersConfig":
        if STAFF_ROLE not in self.roles:
            raise ValueError(f"roles must include {STAFF_ROLE!r}, the role of every user")
        for role in self.roles:
            if not re.match(r"^[a-z][a-z_]*$", role):
                raise ValueError(f"role {role!r} must be lower-case letters and _")
        for login, entry in self.users.items():
            if not re.match(r"^[a-z][a-z0-9._-]{1,31}$", login):
                raise ValueError(f"login {login!r} must be 2–32 lower-case letters, digits, . _ -")
            unknown = sorted(set(entry.roles) - set(self.roles))
            if unknown:
                raise ValueError(f"user {login}: unknown roles {', '.join(unknown)}")
        return self

    def user(self, login: str) -> User | None:
        entry = self.users.get(login)
        if entry is None:
            return None
        roles = tuple(sorted({*entry.roles, STAFF_ROLE}))
        return User(login=login, name=entry.name, title=entry.title, roles=roles)


# --- loading ---

# POLICY_ variables that Docker Compose reads rather than the app.
_COMPOSE_ONLY = {"POLICY_ASSISTANT_HOME"}


def load_settings(**overrides: Any) -> Settings:
    _check_variable_names()
    try:
        return Settings(**overrides)
    except ValidationError as error:
        raise ConfigError(f"settings: {describe_validation_error(error)}") from None


def load_models(path: Path) -> ModelsConfig:
    return _load_yaml(path, ModelsConfig)


def load_users(path: Path) -> UsersConfig:
    return _load_yaml(path, UsersConfig)


def describe_validation_error(error: ValidationError) -> str:
    problems = []
    for problem in error.errors()[:3]:
        where = ".".join(str(part) for part in problem["loc"])
        message = problem["msg"].removeprefix("Value error, ")
        problems.append(f"{where}: {message}" if where else message)
    return "; ".join(problems)


def _load_yaml(path: Path, model: type[BaseModel]):
    if not path.is_file():
        raise ConfigError(f"{path} is missing")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: not valid YAML: {error}") from None
    try:
        return model.model_validate(data)
    except ValidationError as error:
        raise ConfigError(f"{path}: {describe_validation_error(error)}") from None


def _check_variable_names() -> None:
    known = {f"POLICY_{name.upper()}" for name in Settings.model_fields} | _COMPOSE_ONLY
    names = {name for name in os.environ if name.startswith("POLICY_")}
    names |= {name for name in read_env_file(ENV_FILE) if name.startswith("POLICY_")}
    unknown = sorted(names - known)
    if unknown:
        raise ConfigError(f"unknown settings {', '.join(unknown)}; see .env.example for the names")


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines of a .env file; comments and blank lines are skipped."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values
