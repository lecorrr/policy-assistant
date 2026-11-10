"""Shared fixtures.

Database tests need a Postgres server with pgvector and set PA_TEST_DB=1; they then create a
throwaway database and reader role with the admin account from the settings (.env), and drop
both afterwards. Without PA_TEST_DB they are skipped.
"""

import os
import secrets
import textwrap
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from policy_assistant.config import PROJECT_ROOT, Settings, load_models, load_users
from policy_assistant.contracts import Hit
from policy_assistant.text import Lemmatizer

CORPUS_DIR = PROJECT_ROOT / "corpus" / "policies"
USERS_FILE = PROJECT_ROOT / "config" / "users.yaml"
MODELS_FILE = PROJECT_ROOT / "config" / "models.yaml"


@pytest.fixture(scope="session")
def lemmatizer() -> Lemmatizer:
    return Lemmatizer()


@pytest.fixture(scope="session")
def users():
    return load_users(USERS_FILE)


@pytest.fixture(scope="session")
def models():
    return load_models(MODELS_FILE)


def make_settings(**values) -> Settings:
    """Settings that never read the developer's .env."""
    defaults = {"session_secret": "s" * 40}
    return Settings(_env_file=None, **(defaults | values))


# --- an invented law in the official text's format ---

SAMPLE_LAW = textwrap.dedent(
    """\
    ЗАКОН УКРАЇНИ
    Про споживче кредитування
    (Відомості Верховної Ради (ВВР), 2017, № 1, ст.2)
    {Із змінами, внесеними згідно із Законами
    № 122-IX від 20.09.2019}
    Цей Закон визначає правила для прикладу.
    Розділ I
    ЗАГАЛЬНІ ПОЛОЖЕННЯ
    Стаття 1. Визначення термінів
    1. У цьому Законі терміни вживаються в такому значенні:
    1) споживач - фізична особа, яка уклала договір;
    {Пункт 1 частини першої статті 1 із змінами}
    2. Інші терміни вживаються у значенні іншого закону.
    21. Вставлена частина два-один.
    Стаття 2. Мета
    Метою цього Закону є захист споживачів.
    Розділ III1
    ВРЕГУЛЮВАННЯ ПРОСТРОЧЕНОЇ ЗАБОРГОВАНОСТІ
    Стаття 3. Взаємодія
    1. Забороняється взаємодіяти у період з 20 до 9 години.
    7. Цей рядок з номером не по порядку лишається текстом частини 1.
    Розділ IV
    ПРИКІНЦЕВІ ТА ПЕРЕХІДНІ ПОЛОЖЕННЯ
    1. Цей Закон набирає чинності через шість місяців.
    2. Внести зміни до таких законодавчих актів:
    "Стаття 5. Цитата з іншого закону, а не стаття цього.
    1. Цитована частина.
    6. Ще один рядок цитати з номером не по порядку.
    62. Вставлений пункт шість-два (не по порядку, бо 6 немає).
    3. Третій пункт.
    31. Пункт три-один.
    Президент України
    П.ПОРОШЕНКО
    м. Київ
    Додаток 1
    до Закону України
    "Про споживче кредитування"
    Паспорт споживчого кредиту
    1. Інформація та контактні дані кредитодавця
    """
)


# --- an in-memory store that filters by roles the way the database does ---


class MemoryStore:
    """Holds hits and returns those whose roles overlap the user's, best word overlap first."""

    def __init__(self, hits: list[Hit], lemmatizer: Lemmatizer) -> None:
        self.hits = hits
        self._lemmatizer = lemmatizer
        self.calls: list[tuple] = []

    def search(self, roles, *, vector, tsquery, limit):
        self.calls.append((tuple(roles), vector is not None, tsquery))
        readable = [hit for hit in self.hits if set(hit.roles) & set(roles)]
        wanted = set()
        if tsquery:
            wanted = {term.strip("' ") for term in tsquery.split("|")}

        def overlap(hit: Hit) -> int:
            return len(wanted & set(self._lemmatizer.lexemes(hit.search_text)))

        lexical = sorted((h for h in readable if overlap(h)), key=lambda h: -overlap(h))[:limit]
        vectors = readable[:limit] if vector is not None else []
        return vectors, lexical

    def documents(self, roles):
        seen = {}
        for hit in self.hits:
            if set(hit.roles) & set(roles):
                seen[hit.doc_id] = {
                    "id": hit.doc_id,
                    "title": hit.doc_id,
                    "kind": "policy",
                    "version": "1",
                    "url": None,
                }
        return list(seen.values())

    def ping(self) -> None:
        return None


def hit(chunk_id: int, doc_id: str, body: str, roles: tuple[str, ...], section: str = "1") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section=section,
        label=f"{doc_id}, розд. {section}",
        heading=f"{doc_id}. {section}.",
        body=body,
        roles=roles,
        url=None,
    )


SAMPLE_HITS = [
    hit(
        1,
        "staff-handbook",
        "Заяву на відпустку подавайте щонайменше за 14 календарних днів.",
        ("staff",),
    ),
    hit(2, "hr-salary-bands", "G2, менеджер з продажу: від 28 000 до 38 000 грн.", ("hr",)),
    hit(
        3,
        "sales-discounts",
        "Менеджер може знизити ставку не більше ніж на 2 відсоткові пункти.",
        ("sales",),
    ),
    hit(4, "risk-credit-policy", "Заявку з балом 640 і вище схвалюють автоматично.", ("risk",)),
]


@pytest.fixture
def memory_store(lemmatizer) -> MemoryStore:
    return MemoryStore(list(SAMPLE_HITS), lemmatizer)


# --- a real database for the integration tests ---


@pytest.fixture(scope="session")
def database():
    """Connection strings and POLICY_ variables for a throwaway database, or a skip."""
    if os.environ.get("PA_TEST_DB") != "1":
        pytest.skip("set PA_TEST_DB=1 to run the database tests")
    settings = Settings()
    base = conninfo_to_dict(settings.admin_dsn)
    name = f"policy_test_{secrets.token_hex(4)}"
    reader = f"policy_reader_{secrets.token_hex(4)}"
    reader_password = secrets.token_urlsafe(16)
    with psycopg.connect(settings.admin_dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    admin_dsn = make_conninfo(**(base | {"dbname": name}))
    reader_dsn = make_conninfo(
        **(base | {"dbname": name, "user": reader, "password": reader_password})
    )
    # The same database for commands, which build their connection strings from settings.
    env = {
        "POLICY_DB_NAME": name,
        "POLICY_DB_READER_USER": reader,
        "POLICY_DB_READER_PASSWORD": reader_password,
    }
    try:
        yield {"admin": admin_dsn, "reader": reader_dsn, "env": env}
    finally:
        with psycopg.connect(settings.admin_dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )
            conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(reader)))


def law_text_path() -> Path | None:
    """The downloaded law, if this machine has it (`fetch-law`)."""
    path = Settings(_env_file=None).law_file
    return path if path.is_file() else None
