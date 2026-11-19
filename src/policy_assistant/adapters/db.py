import math
import re
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

from policy_assistant.contracts import ChunkDraft, Document, Hit

_ROLE = re.compile(r"^[a-z][a-z_]*$")
_START_HINT = "is Postgres running? `docker compose up -d --wait db` starts it"


class DatabaseError(RuntimeError):
    pass


def roles_literal(roles: Iterable[str]) -> str:
    """Build the text[] literal for app.roles; only plain role names are allowed."""
    names = sorted(set(roles))
    for name in names:
        if not _ROLE.match(name):
            raise ValueError(f"not a role name: {name!r}")
    return "{" + ",".join(names) + "}"


def vector_literal(values: Sequence[float]) -> str:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("an embedding contains NaN or infinity")
    return "[" + ",".join(repr(float(value)) for value in values) + "]"


@dataclass(frozen=True)
class StoredChunk:
    draft: ChunkDraft
    lexemes: str  # a tsvector literal
    embedding: str  # a vector literal
    content_hash: str


@dataclass(frozen=True)
class ReaderCheck:
    pgvector_version: str
    reader_is_superuser: bool
    reader_bypasses_rls: bool
    reader_owns_tables: bool
    rls_enabled: bool
    rows_without_roles: int  # what the reader sees with no roles set: must be 0
    chunks_by_role: dict[str, int]

    @property
    def problems(self) -> list[str]:
        found = []
        if self.reader_is_superuser:
            found.append("the reader role is a superuser, so row-level security does not apply")
        if self.reader_bypasses_rls:
            found.append("the reader role has BYPASSRLS")
        if self.reader_owns_tables:
            found.append("the reader role owns the tables, so row-level security does not apply")
        if not self.rls_enabled:
            found.append("row-level security is not enabled on documents and chunks")
        if self.rows_without_roles:
            found.append(f"the reader sees {self.rows_without_roles} chunks with no roles set")
        return found


def connect_admin(dsn: str) -> psycopg.Connection:
    try:
        return psycopg.connect(dsn, connect_timeout=10, row_factory=dict_row)
    except psycopg.OperationalError as error:
        raise DatabaseError(
            f"cannot connect as the admin: {_first_line(error)}; {_START_HINT}"
        ) from None


def create_schema(
    admin_dsn: str, reader_dsn: str, *, dims: int, embedding_model: str, reset: bool = False
) -> None:
    """Create or update the schema, the reader role and its grants. Safe to re-run."""
    reader = conninfo_to_dict(reader_dsn)
    reader_user, reader_password = reader.get("user"), reader.get("password")
    if not reader_user or not reader_password:
        raise DatabaseError(
            "the reader's user and password must be set (POLICY_DB_READER_PASSWORD)"
        )
    with connect_admin(admin_dsn) as conn:
        if reset:
            conn.execute("DROP TABLE IF EXISTS chunks, documents, schema_info")
        stored = _stored_dims(conn)
        if stored is not None and stored != dims:
            raise DatabaseError(
                f"the database holds {stored}-dimension embeddings but POLICY_EMBEDDING_DIMS is "
                f"{dims}: run `policy-assistant init-db --reset`, then `ingest`"
            )
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(_SCHEMA.format(dims=int(dims)))
        conn.execute(
            "INSERT INTO schema_info (key, value) VALUES ('embedding_dims', %s) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            [str(dims)],
        )
        _ensure_reader(conn, reader_user, reader_password)
        conn.commit()


def check_embedding_setup(admin_dsn: str, *, dims: int) -> None:
    with connect_admin(admin_dsn) as conn:
        stored = _stored_dims(conn)
    if stored is None:
        raise DatabaseError("the schema is missing: run `policy-assistant init-db` first")
    if stored != dims:
        raise DatabaseError(
            f"the database holds {stored}-dimension embeddings, the embedder makes {dims}: "
            "run `policy-assistant init-db --reset`, then `ingest`"
        )


def known_embeddings(admin_dsn: str, content_hashes: list[str]) -> dict[str, str]:
    """Stored embeddings for these content hashes, so ingest can skip them."""
    with connect_admin(admin_dsn) as conn:
        rows = conn.execute(
            "SELECT DISTINCT ON (content_hash) content_hash, embedding::text AS embedding "
            "FROM chunks WHERE content_hash = ANY(%s)",
            [content_hashes],
        ).fetchall()
    return {row["content_hash"]: row["embedding"] for row in rows}


def replace_corpus(
    admin_dsn: str, documents: list[Document], chunks: list[StoredChunk], *, embedding_model: str
) -> None:
    """Replace the corpus in one transaction: readers see old or new, never a mix."""
    ids = [document.id for document in documents]
    with connect_admin(admin_dsn) as conn, conn.transaction():
        conn.execute(
            "INSERT INTO schema_info (key, value) VALUES ('embedding_model', %s) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            [embedding_model],
        )
        conn.execute("DELETE FROM documents WHERE NOT (id = ANY(%s))", [ids])
        for document in documents:
            conn.execute(
                "INSERT INTO documents (id, title, kind, source, version, url, allowed_roles) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET "
                "title = excluded.title, kind = excluded.kind, source = excluded.source, "
                "version = excluded.version, url = excluded.url, "
                "allowed_roles = excluded.allowed_roles, ingested_at = now()",
                [
                    document.id,
                    document.title,
                    document.kind,
                    document.source,
                    document.version,
                    document.url,
                    list(document.roles),
                ],
            )
        conn.execute("DELETE FROM chunks WHERE document_id = ANY(%s)", [ids])
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO chunks (document_id, position, section, label, heading, body, "
                "allowed_roles, lexemes, embedding, content_hash) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::tsvector, %s::vector, %s)",
                [
                    (
                        c.draft.doc_id,
                        c.draft.position,
                        c.draft.section,
                        c.draft.label,
                        c.draft.heading,
                        c.draft.body,
                        list(c.draft.roles),
                        c.lexemes,
                        c.embedding,
                        c.content_hash,
                    )
                    for c in chunks
                ],
            )


def check_reader(admin_dsn: str, reader_dsn: str) -> ReaderCheck:
    reader_user = conninfo_to_dict(reader_dsn).get("user")
    with connect_admin(admin_dsn) as conn:
        role = conn.execute(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = %s", [reader_user]
        ).fetchone()
        if role is None:
            raise DatabaseError(f"the reader role {reader_user!r} is missing: run `init-db`")
        tables = conn.execute(
            "SELECT c.relname, pg_get_userbyid(c.relowner) AS owner, c.relrowsecurity "
            "FROM pg_class c WHERE c.relname IN ('documents', 'chunks') AND c.relkind = 'r'"
        ).fetchall()
        if len(tables) != 2:
            raise DatabaseError("the tables are missing: run `policy-assistant init-db`")
        version = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        by_role = conn.execute(
            "SELECT role, count(*) AS n FROM chunks, unnest(allowed_roles) AS role "
            "GROUP BY role ORDER BY role"
        ).fetchall()
    with _connect_reader(reader_dsn) as conn:
        unscoped = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    return ReaderCheck(
        pgvector_version=version["extversion"] if version else "missing",
        reader_is_superuser=role["rolsuper"],
        reader_bypasses_rls=role["rolbypassrls"],
        reader_owns_tables=any(t["owner"] == reader_user for t in tables),
        rls_enabled=all(t["relrowsecurity"] for t in tables),
        rows_without_roles=unscoped,
        chunks_by_role={row["role"]: row["n"] for row in by_role},
    )


class AdminStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def known_embeddings(self, content_hashes: list[str]) -> dict[str, str]:
        return known_embeddings(self.dsn, content_hashes)

    def replace_corpus(
        self, documents: list[Document], chunks: list[StoredChunk], *, embedding_model: str
    ) -> None:
        replace_corpus(self.dsn, documents, chunks, embedding_model=embedding_model)


class ReaderStore:
    """Searches as the reader role, in a transaction scoped to one user's roles."""

    def __init__(self, dsn: str, *, max_size: int = 4, timeout_s: float = 10) -> None:
        self._pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=max_size,
            timeout=timeout_s,
            open=False,
            kwargs={"row_factory": dict_row, "connect_timeout": int(timeout_s)},
            configure=_load_pgvector,
        )
        try:
            self._pool.open(wait=True, timeout=timeout_s)
        except PoolTimeout:
            self._pool.close()
            raise DatabaseError(f"cannot connect as the reader; {_START_HINT}") from None

    def close(self) -> None:
        self._pool.close()

    def search(
        self,
        roles: Iterable[str],
        *,
        vector: Sequence[float] | None,
        tsquery: str | None,
        limit: int,
    ) -> tuple[list[Hit], list[Hit]]:
        """Return (vector hits, full-text hits), best first, limited to the user's roles."""
        vector_hits: list[Hit] = []
        lexical_hits: list[Hit] = []
        with self._scoped(roles) as conn:
            # RLS filters after the HNSW scan; iterative scan keeps going until `limit` rows pass.
            conn.execute(
                "SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true), "
                "set_config('hnsw.ef_search', %s, true)",
                [str(max(40, 2 * limit))],
            )
            if vector is not None:
                rows = conn.execute(
                    _VECTOR_SQL, {"v": vector_literal(vector), "n": limit}
                ).fetchall()
                vector_hits = [_hit(row, 1.0 - row["distance"]) for row in rows]
            if tsquery is not None:
                rows = conn.execute(_LEXICAL_SQL, {"q": tsquery, "n": limit}).fetchall()
                lexical_hits = [_hit(row, row["rank"]) for row in rows]
        return vector_hits, lexical_hits

    def documents(self, roles: Iterable[str]) -> list[dict]:
        with self._scoped(roles) as conn:
            return conn.execute(
                "SELECT id, title, kind, version, url FROM documents ORDER BY kind, title"
            ).fetchall()

    def ping(self) -> None:
        with self._scoped([]) as conn:
            conn.execute("SELECT 1")

    def embedding_model(self) -> str | None:
        with self._scoped([]) as conn:
            row = conn.execute(
                "SELECT value FROM schema_info WHERE key = 'embedding_model'"
            ).fetchone()
        return row["value"] if row else None

    @contextmanager
    def _scoped(self, roles: Iterable[str]) -> Iterator[psycopg.Connection]:
        literal = roles_literal(roles)
        try:
            with self._pool.connection() as conn, conn.transaction():
                # true: local to this transaction
                conn.execute("SELECT set_config('app.roles', %s, true)", [literal])
                yield conn
        except PoolTimeout:
            raise DatabaseError(f"no database connection became free; {_START_HINT}") from None
        except psycopg.OperationalError as error:
            raise DatabaseError(f"the database connection failed: {_first_line(error)}") from None


def _connect_reader(dsn: str) -> psycopg.Connection:
    try:
        return psycopg.connect(dsn, connect_timeout=10, row_factory=dict_row)
    except psycopg.OperationalError as error:
        raise DatabaseError(f"cannot connect as the reader: {_first_line(error)}") from None


def _load_pgvector(conn: psycopg.Connection) -> None:
    # Load pgvector first so set_config('hnsw.*') sets the real setting.
    conn.execute("SELECT '[1]'::vector")
    conn.commit()


def _hit(row: dict, score: float) -> Hit:
    return Hit(
        chunk_id=row["id"],
        doc_id=row["document_id"],
        section=row["section"],
        label=row["label"],
        heading=row["heading"],
        body=row["body"],
        roles=tuple(row["allowed_roles"]),
        url=row["url"],
        score=float(score),
    )


def _stored_dims(conn: psycopg.Connection) -> int | None:
    exists = conn.execute("SELECT to_regclass('schema_info') IS NOT NULL AS ok").fetchone()["ok"]
    if not exists:
        return None
    row = conn.execute("SELECT value FROM schema_info WHERE key = 'embedding_dims'").fetchone()
    return int(row["value"]) if row else None


def _ensure_reader(conn: psycopg.Connection, user: str, password: str) -> None:
    exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", [user]).fetchone()
    name = sql.Identifier(user)
    if exists is None:
        conn.execute(
            sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE").format(
                name
            )
        )
    conn.execute(
        sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(name, sql.Literal(password))
    )
    database = conn.execute("SELECT current_database() AS db").fetchone()["db"]
    conn.execute(
        sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(database), name)
    )
    conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(name))
    conn.execute(sql.SQL("GRANT SELECT ON documents, chunks, schema_info TO {}").format(name))


def _first_line(error: Exception) -> str:
    return str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (key text PRIMARY KEY, value text NOT NULL);

CREATE TABLE IF NOT EXISTS documents (
    id text PRIMARY KEY,
    title text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('law', 'policy')),
    source text NOT NULL,
    version text NOT NULL,
    url text,
    allowed_roles text[] NOT NULL CHECK (cardinality(allowed_roles) > 0),
    ingested_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id text NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    position integer NOT NULL,
    section text NOT NULL,
    label text NOT NULL,
    heading text NOT NULL,
    body text NOT NULL,
    allowed_roles text[] NOT NULL CHECK (cardinality(allowed_roles) > 0),
    lexemes tsvector NOT NULL,
    embedding vector({dims}) NOT NULL,
    content_hash text NOT NULL,
    UNIQUE (document_id, position)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_lexemes_idx ON chunks USING gin (lexemes);
CREATE INDEX IF NOT EXISTS chunks_hash_idx ON chunks (content_hash);

-- Empty, never NULL, when app.roles is unset: the policies below then match no rows.
CREATE OR REPLACE FUNCTION app_current_roles() RETURNS text[]
LANGUAGE sql STABLE AS $$
    SELECT coalesce(nullif(current_setting('app.roles', true), ''), '{{}}')::text[]
$$;

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS documents_by_role ON documents;
CREATE POLICY documents_by_role ON documents FOR SELECT
    USING (allowed_roles && app_current_roles());
DROP POLICY IF EXISTS chunks_by_role ON chunks;
CREATE POLICY chunks_by_role ON chunks FOR SELECT
    USING (allowed_roles && app_current_roles());
"""

_VECTOR_SQL = """
SELECT hit.*, d.url FROM (
    SELECT id, document_id, section, label, heading, body, allowed_roles,
           embedding <=> %(v)s::vector AS distance
    FROM chunks
    ORDER BY embedding <=> %(v)s::vector
    LIMIT %(n)s
) AS hit JOIN documents d ON d.id = hit.document_id
ORDER BY hit.distance, hit.id
"""

# Headings count double: on the 48 answerable questions this gave 92 % top-5 recall,
# vs 83 % with Postgres' default 10x heading weight and 77 % with ts_rank_cd.
_LEXICAL_SQL = """
SELECT hit.*, d.url FROM (
    SELECT id, document_id, section, label, heading, body, allowed_roles,
           ts_rank('{0.1, 0.2, 0.4, 0.2}', lexemes, query) AS rank
    FROM chunks, CAST(%(q)s AS tsquery) AS query
    WHERE lexemes @@ query
    ORDER BY rank DESC, id
    LIMIT %(n)s
) AS hit JOIN documents d ON d.id = hit.document_id
ORDER BY hit.rank DESC, hit.id
"""
