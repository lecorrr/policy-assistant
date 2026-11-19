import pytest

from policy_assistant.ranking import reciprocal_rank_fusion
from policy_assistant.text import normalize, numbers_in, tsquery_literal, tsvector_literal


def test_normalize_unifies_case_apostrophes_quotes_dashes_and_spaces():
    assert (
        normalize("Зобов’язання «ДемоКредит» — 28 000  грн")
        == 'зобов\'язання "демокредит" - 28 000 грн'
    )
    assert normalize("пов'язану") == normalize("повʼязану")


def test_numbers_in_reads_thousands_and_decimal_commas():
    assert numbers_in("від 28 000 до 38 000 грн, знижка 1,5 в.п. і 640 балів") == {
        "28000",
        "38000",
        "1.5",
        "640",
    }
    assert numbers_in("без цифр") == set()


def test_lexemes_lemmatise_and_drop_stop_words(lemmatizer):
    assert lemmatizer.lexemes("Скільки днів має клієнт для відмови?") == [
        "день",
        "клієнт",
        "відмова",
    ]


def test_every_reading_of_ambiguous_word_is_kept(lemmatizer):
    # The dictionary has no frequencies: "пені" may be "пеня" or "пен".
    assert {"пеня", "пен"} <= set(lemmatizer.lexemes("пені"))
    assert "пеня" in lemmatizer.lexemes("пеня")


def test_apostrophes_and_hyphens(lemmatizer):
    assert (
        lemmatizer.lexemes("зобов'язання") == lemmatizer.lexemes("зобов’язання") == ["зобовязання"]
    )
    terms = lemmatizer.lexemes("ризик-менеджер")
    assert "ризикменеджер" in terms and "менеджер" in terms


def test_latin_words_and_digits_pass_through(lemmatizer):
    assert lemmatizer.lexemes("CRM 0142 fraud@demokredyt.example") == [
        "crm",
        "0142",
        "fraud",
        "demokredyt",
        "example",
    ]


@pytest.mark.parametrize(
    "text", ["'; DROP TABLE chunks; --", "a & b | !c", "назва\\'з\\:двокрапкою", "{}[]()<>"]
)
def test_terms_cannot_break_out_of_literal(lemmatizer, text):
    for term in lemmatizer.lexemes(text):
        assert set(term) <= set(
            "abcdefghijklmnopqrstuvwxyz0123456789абвгґдеєжзиіїйклмнопрстуфхцчшщьюя-"
        )
    literal = tsquery_literal(lemmatizer.lexemes(text))
    assert literal is None or literal.count("'") % 2 == 0


def test_tsvector_positions_weights_and_shared_positions():
    literal = tsvector_literal([["стаття"]], [["пеня", "пен"], ["стаття"]])
    parts = dict(item.split(":", 1) for item in literal.split())
    assert parts["'стаття'"] == "1A,3"
    assert parts["'пеня'"] == parts["'пен'"] == "2"


def test_tsquery_is_none_without_terms_and_deduplicates():
    assert tsquery_literal([]) is None
    assert tsquery_literal(["кредит", "строк", "кредит"]) == "'кредит' | 'строк'"


def test_reciprocal_rank_fusion_rewards_agreement_and_is_deterministic():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    order = [item for item, _ in fused]
    assert order[:2] == [1, 3]  # in both lists, near the top
    assert set(order) == {1, 2, 3, 4}
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert reciprocal_rank_fusion([[5, 6], [6, 5]]) == reciprocal_rank_fusion([[5, 6], [6, 5]])
    assert [i for i, _ in reciprocal_rank_fusion([[5, 6], [6, 5]])] == [5, 6]  # tie: first seen


def test_fusion_of_nothing_is_empty():
    assert reciprocal_rank_fusion([[], []]) == []
