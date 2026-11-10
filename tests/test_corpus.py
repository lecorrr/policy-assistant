"""Parsing the law and the policies, and chunking them."""

import textwrap

import pytest
from conftest import CORPUS_DIR, SAMPLE_LAW, law_text_path

from policy_assistant.contracts import STAFF_ROLE, Block, Document, Paragraph
from policy_assistant.corpus import CorpusError, load_law, load_policies
from policy_assistant.corpus.chunking import chunk_document, chunk_documents
from policy_assistant.corpus.law import LawFormatError, law_label, parse_law, read_law_bytes
from policy_assistant.corpus.policies import PolicyFormatError, parse_policy

ROLES = {"staff", "sales", "risk", "hr"}


def parse_sample(text: str = SAMPLE_LAW):
    return parse_law(text, source="test", version="test")


# --- the law ---


def test_law_blocks_follow_the_official_structure():
    document = parse_sample()
    assert [block.section for block in document.blocks] == [
        "преамбула",
        "1",
        "2",
        "3",
        "IV.1",
        "IV.2",
        "IV.3",
        "IV.3-1",
        "Додаток 1",
    ]
    assert document.blocks[1].title == "Стаття 1. Визначення термінів"
    assert document.blocks[-1].title == "Додаток 1. Паспорт споживчого кредиту"
    assert all(block.roles == (STAFF_ROLE,) for block in document.blocks)


def test_editorial_notes_are_dropped_even_across_lines():
    text = " ".join(p.text for block in parse_sample().blocks for p in block.paragraphs)
    assert "{" not in text and "122-IX" not in text and "Пункт 1 частини першої" not in text


def test_parts_are_numbered_in_sequence_with_flattened_inserts():
    article = parse_sample().blocks[1]
    assert [p.part for p in article.paragraphs] == ["1", "1", "2", "2-1"]
    # A number out of sequence is text of the current part, not a new part.
    third = parse_sample().blocks[3]
    assert [p.part for p in third.paragraphs] == ["1", "1"]


def test_quoted_amendments_in_the_final_provisions_stay_text():
    point = next(b for b in parse_sample().blocks if b.section == "IV.2")
    assert len(point.paragraphs) == 5  # the point, the quoted article and three quoted lines
    assert not any(b.title.startswith("Стаття 5") for b in parse_sample().blocks)


def test_section_with_flattened_superscript_and_title_on_the_next_line():
    point = next(b for b in parse_sample().blocks if b.section == "IV.3-1")
    assert point.title == "Розділ IV. Прикінцеві та перехідні положення, пункт 3-1"


def test_hyphenated_numbers_from_the_website_are_read_too():
    text = SAMPLE_LAW.replace("Розділ III1", "Розділ III-1").replace("31. Пункт", "3-1. Пункт")
    sections = [b.section for b in parse_sample(text).blocks]
    assert "IV.3-1" in sections


def test_windows_line_ends_and_cp1251_decode():
    raw = SAMPLE_LAW.replace("\n", "\r\n").encode("cp1251")
    assert [b.section for b in parse_sample(read_law_bytes(raw)).blocks] == [
        b.section for b in parse_sample().blocks
    ]


@pytest.mark.parametrize(
    "broken, message",
    [
        (SAMPLE_LAW.replace("№ 122-IX від 20.09.2019}", "№ 122-IX від 20.09.2019"), "never closed"),
        (SAMPLE_LAW.replace("Стаття 2. Мета", "} Стаття 2. Мета"), "closing brace"),
        ("Просто текст\nбез статей", "no articles"),
        (
            SAMPLE_LAW.replace("Стаття 1. Визначення термінів", "Вступ без статті"),
            "before the first article",
        ),
    ],
)
def test_text_that_is_not_the_law_is_refused(broken, message):
    with pytest.raises(LawFormatError, match=message):
        parse_sample(broken)


def test_law_labels():
    assert law_label("15", []) == "Закон № 1734-VIII, ст. 15"
    assert law_label("15", ["1", "3"]) == "Закон № 1734-VIII, ст. 15, ч. 1–3"
    assert law_label("15", ["2"]) == "Закон № 1734-VIII, ст. 15, ч. 2"
    assert law_label("IV.6-2", []) == "Закон № 1734-VIII, розд. IV, п. 6-2"
    assert law_label("Додаток 1", []) == "Закон № 1734-VIII, додаток 1"


def test_a_missing_law_file_says_how_to_get_it(tmp_path):
    with pytest.raises(CorpusError, match="fetch-law"):
        load_law(tmp_path / "1734-19.txt")


@pytest.mark.skipif(law_text_path() is None, reason="the law's text is not downloaded")
def test_the_downloaded_law_has_all_28_articles():
    document = load_law(law_text_path())
    articles = [b.section for b in document.blocks if b.title.startswith("Стаття")]
    assert articles == [str(n) for n in range(1, 29)]
    assert any(b.section.startswith("IV.") for b in document.blocks)


# --- the policies ---

POLICY = textwrap.dedent(
    """\
    ---
    id: sample
    title: Зразок політики
    roles: [sales]
    owner: Відділ
    version: "1.0"
    effective: 2026-01-01
    ---

    > **ЗРАЗОК.** Вигаданий документ.

    ## 1. Перший розділ

    Перший абзац.
    Він має два рядки.

    Другий абзац.

    ## 2. Для відділу персоналу
    <!-- roles: hr -->

    Лише для HR.
    """
)


def write_policy(tmp_path, text=POLICY, name="sample.md"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_policy_sections_paragraphs_and_role_override(tmp_path):
    document = parse_policy(write_policy(tmp_path), ROLES)
    first, second = document.blocks
    assert (first.section, first.title, first.roles) == ("1", "1. Перший розділ", ("sales",))
    assert [p.text for p in first.paragraphs] == [
        "Перший абзац.\nВін має два рядки.",
        "Другий абзац.",
    ]
    assert second.roles == ("hr",)
    assert document.roles == ("hr", "sales")
    assert "ЗРАЗОК" not in " ".join(p.text for b in document.blocks for p in b.paragraphs)


@pytest.mark.parametrize(
    "change, message",
    [
        (("roles: [sales]", "roles: [sales, boss]"), "unknown roles boss"),
        (("roles: [sales]", "roles: []"), "roles"),
        (("owner: Відділ\n", "owner: Відділ\ncolour: red\n"), "colour"),
        (("id: sample", "id: other"), "differs from the file name"),
        (("## 2. Для відділу персоналу", "## Для відділу персоналу"), "headings must look like"),
        (("## 2. Для відділу персоналу", "## 1. Знову перший"), "appears twice"),
        (("<!-- roles: hr -->", "<!-- roles: boss -->"), "unknown roles boss"),
        (("Другий абзац.", "Другий абзац.\n<!-- roles: hr -->"), "on the line under a heading"),
        (("Лише для HR.", ""), "is empty"),
        (("---\nid: sample", "id: sample"), "front matter"),
    ],
)
def test_policy_format_errors_name_the_problem(tmp_path, change, message):
    old, new = change
    with pytest.raises(PolicyFormatError, match=message):
        parse_policy(write_policy(tmp_path, POLICY.replace(old, new, 1)), ROLES)


def test_every_shipped_policy_parses_and_is_stamped_as_a_sample():
    documents = load_policies(CORPUS_DIR, ROLES)
    assert len(documents) == 15
    for path in CORPUS_DIR.glob("*.md"):
        assert "**ЗРАЗОК.**" in path.read_text(encoding="utf-8"), path.name
    roles = {role for d in documents for role in d.roles}
    assert roles == ROLES


# --- chunking ---


def law_document(paragraphs: list[Paragraph]) -> Document:
    block = Block("9", "Стаття 9. Інформація", tuple(paragraphs), ("staff",))
    return Document("law-1734", "Закон", "law", "test", "test", (block,))


def test_a_short_block_is_one_chunk_labelled_without_parts():
    document = law_document([Paragraph("Перша частина.", "1"), Paragraph("Друга частина.", "2")])
    (chunk,) = chunk_document(document)
    assert chunk.label == "Закон № 1734-VIII, ст. 9"
    assert chunk.body == "Перша частина.\nДруга частина."
    assert chunk.heading == "Закон. Стаття 9. Інформація"


def test_a_long_block_splits_on_paragraphs_and_names_its_parts():
    paragraphs = [Paragraph(f"Частина {n}. " + "слово " * 60, str(n)) for n in range(1, 7)]
    chunks = chunk_document(law_document(paragraphs), max_chars=800)
    assert len(chunks) > 1
    assert all(len(chunk.body) <= 800 * 1.3 for chunk in chunks)
    assert chunks[0].label.startswith("Закон № 1734-VIII, ст. 9, ч. 1")
    assert " ".join(c.body for c in chunks).count("Частина") == 6  # nothing lost or doubled
    assert len({c.label for c in chunks}) == len(chunks)


def test_one_huge_paragraph_is_cut_at_sentences_and_numbered():
    text = " ".join(f"Речення номер {n} про щось важливе." for n in range(200))
    chunks = chunk_document(law_document([Paragraph(text, "1")]), max_chars=500)
    assert len(chunks) > 5
    assert all(c.body.endswith(".") for c in chunks)
    assert chunks[0].label.endswith("фрагмент 1/" + str(len(chunks)))


def test_chunks_never_span_blocks_and_carry_block_roles(tmp_path):
    document = parse_policy(write_policy(tmp_path), ROLES)
    chunks = chunk_documents([document])
    assert [(c.section, c.roles) for c in chunks] == [("1", ("sales",)), ("2", ("hr",))]
    assert chunks[1].label == "Зразок політики, розд. 2"


def test_tiny_chunk_size_is_refused():
    with pytest.raises(ValueError, match="200"):
        chunk_document(law_document([Paragraph("x", "1")]), max_chars=100)
