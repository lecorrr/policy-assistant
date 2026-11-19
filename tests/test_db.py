import psycopg
import pytest
from conftest import CORPUS_DIR, SAMPLE_LAW

from policy_assistant.adapters import db
from policy_assistant.corpus import load_corpus
from policy_assistant.fakes import HashingEmbedder, OverlapReranker
from policy_assistant.ingest import ingest
from policy_assistant.search import AccessViolation, Retriever
from policy_assistant.text import Lemmatizer, tsquery_literal

DIMS = 64
ROLES = {"staff", "sales", "risk", "hr"}


@pytest.fixture(scope="module")
def stored(database, tmp_path_factory):
    lemmatizer = Lemmatizer()
    db.create_schema(
        database["admin"], database["reader"], dims=DIMS, embedding_model="fake", reset=True
    )
    db.create_schema(
        database["admin"], database["reader"], dims=DIMS, embedding_model="fake"
    )  # again
    law = tmp_path_factory.mktemp("law") / "1734-19.txt"
    law.write_text(SAMPLE_LAW, encoding="utf-8")
    documents = load_corpus(CORPUS_DIR, law, ROLES)
    embedder = HashingEmbedder(lemmatizer, DIMS)
    report = ingest(
        documents,
        store=db.AdminStore(database["admin"]),
        embedder=embedder,
        embedding_model=embedder.model,
        lemmatizer=lemmatizer,
        max_chars=1200,
    )
    store = db.ReaderStore(database["reader"], max_size=1)
    yield {
        "store": store,
        "report": report,
        "lemmatizer": lemmatizer,
        "embedder": embedder,
        "db": database,
    }
    store.close()


def search(stored, roles, text):
    lemmatizer = stored["lemmatizer"]
    vector = stored["embedder"].embed([text])[0]
    return stored["store"].search(
        roles, vector=vector, tsquery=tsquery_literal(lemmatizer.lexemes(text)), limit=50
    )


def test_reader_is_set_up_safely(stored):
    check = db.check_reader(stored["db"]["admin"], stored["db"]["reader"])
    assert check.problems == []
    assert check.rows_without_roles == 0
    assert check.pgvector_version >= "0.8"
    # 12 chunks in the three HR documents and 4 in the bonus rules HR shares with sales.
    assert check.chunks_by_role["hr"] == 16


def test_hr_text_never_reaches_sales_search_even_verbatim(stored):
    exact = "G2, менеджер з продажу: від 28 000 до 38 000 грн"
    vector_hits, lexical_hits = search(stored, ["staff", "sales"], exact)
    returned = vector_hits + lexical_hits
    assert returned and all(set(h.roles) & {"staff", "sales"} for h in returned)
    assert not any("28 000 до 38 000" in h.body for h in returned)
    hr_hits = search(stored, ["staff", "hr"], exact)[1]
    assert any("28 000 до 38 000" in h.body for h in hr_hits[:3])


def test_section_narrowed_to_hr_is_hidden_inside_shared_document(stored):
    vector_hits, lexical_hits = search(stored, ["staff", "sales"], "код виплати 0142 бонус")
    returned = vector_hits + lexical_hits
    assert any(h.doc_id == "sales-bonus" for h in returned)
    assert not any("0142" in h.body for h in returned)
    vector_hits, lexical_hits = search(stored, ["staff", "hr"], "код виплати 0142")
    assert any("0142" in h.body for h in vector_hits + lexical_hits)


def test_no_roles_means_no_rows(stored):
    assert search(stored, [], "відпустка") == ([], [])
    assert stored["store"].documents([]) == []


def test_documents_are_listed_by_role(stored):
    ids = {d["id"] for d in stored["store"].documents(["staff"])}
    assert "law-1734" in ids and "staff-handbook" in ids
    assert not ids & {"hr-salary-bands", "sales-bonus", "risk-fraud"}


def test_roles_do_not_outlive_request_on_pooled_connection(stored):
    store = stored["store"]  # one connection, reused
    search(stored, ["staff", "hr"], "оклад")
    with store._pool.connection() as conn:
        assert conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"] == 0


def test_reader_cannot_write(stored):
    with psycopg.connect(stored["db"]["reader"]) as conn:
        for statement in (
            "UPDATE chunks SET allowed_roles = '{staff}'",
            "DELETE FROM chunks",
            "INSERT INTO documents (id, title, kind, source, version, allowed_roles) "
            "VALUES ('x', 'x', 'policy', 'x', 'x', '{staff}')",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(statement)
            conn.rollback()


def test_connecting_as_admin_is_caught(stored):
    check = db.check_reader(stored["db"]["admin"], stored["db"]["admin"])
    assert any("superuser" in problem for problem in check.problems)
    as_admin = db.ReaderStore(stored["db"]["admin"])
    try:
        lemmatizer = stored["lemmatizer"]
        retriever = Retriever(as_admin, stored["embedder"], OverlapReranker(lemmatizer), lemmatizer)
        with pytest.raises(AccessViolation, match="reader"):
            retriever.search(("staff",), "оклад менеджера з продажу")
    finally:
        as_admin.close()


def test_ingest_reuses_embeddings_and_records_embedder(stored):
    first = stored["report"]
    assert first.embedded == len(first.chunks) and first.reused == 0
    again = ingest(
        first.documents,
        store=db.AdminStore(stored["db"]["admin"]),
        embedder=stored["embedder"],
        embedding_model=stored["embedder"].model,
        lemmatizer=stored["lemmatizer"],
        max_chars=1200,
    )
    assert again.embedded == 0 and again.reused == len(first.chunks)
    assert stored["store"].embedding_model() == "fake-hashing"


def test_changing_vector_size_needs_reset(stored):
    with pytest.raises(db.DatabaseError, match="init-db --reset"):
        db.create_schema(
            stored["db"]["admin"], stored["db"]["reader"], dims=DIMS * 2, embedding_model="x"
        )
    with pytest.raises(db.DatabaseError, match="init-db --reset"):
        db.check_embedding_setup(stored["db"]["admin"], dims=DIMS * 2)


def test_full_text_search_matches_inflected_forms(stored):
    lexical = search(stored, ["staff", "sales"], "знижок без погодження")[1]
    assert lexical[0].doc_id == "sales-discounts"
