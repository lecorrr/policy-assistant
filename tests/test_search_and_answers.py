"""The retriever and the assistant, on an in-memory store and the stand-in models."""

import pytest
from conftest import MemoryStore, hit

from policy_assistant.adapters.chat import Generation
from policy_assistant.answering import (
    NO_ANSWER,
    REFUSAL,
    SYSTEM_PROMPT,
    Assistant,
    BadQuestion,
    UnknownModel,
    build_user_message,
    parse_answer,
)
from policy_assistant.contracts import Hit, Usage, User
from policy_assistant.fakes import ExtractiveGenerator, HashingEmbedder, OverlapReranker
from policy_assistant.search import AccessViolation, Retriever

OLENA = User("olena", "Олена", "продажі", ("sales", "staff"))
ANDRII = User("andrii", "Андрій", "підтримка", ("staff",))


def retriever(store, lemmatizer, **kwargs) -> Retriever:
    return Retriever(
        store, HashingEmbedder(lemmatizer, 64), OverlapReranker(lemmatizer), lemmatizer, **kwargs
    )


class ScriptedGenerator:
    name = "scripted"

    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> Generation:
        self.prompts.append((system, user))
        return Generation(self.text, Usage(1000, 50), 12.5)


# --- retriever ---


@pytest.mark.parametrize("mode", ["vector", "lexical", "hybrid", "hybrid+rerank"])
def test_every_mode_returns_only_rows_the_user_may_read(memory_store, lemmatizer, mode):
    result = retriever(memory_store, lemmatizer).search(
        OLENA.roles, "знизити ставку відпустка оклад", mode=mode
    )
    assert result.hits
    assert all(set(h.roles) & set(OLENA.roles) for h in result.candidates)
    assert "hr-salary-bands" not in {h.doc_id for h in result.candidates}


def test_vector_only_and_lexical_only_skip_the_other_search(memory_store, lemmatizer):
    search = retriever(memory_store, lemmatizer)
    search.search(OLENA.roles, "ставка", mode="vector")
    search.search(OLENA.roles, "ставка", mode="lexical")
    assert memory_store.calls[0][1:] == (True, None)
    assert memory_store.calls[1][1] is False and memory_store.calls[1][2]


def test_rerank_orders_by_the_reranker_and_keeps_top_k(memory_store, lemmatizer):
    result = retriever(memory_store, lemmatizer, top_k=1).search(
        OLENA.roles, "ставку знизити на відсоткові пункти", mode="hybrid+rerank"
    )
    assert [h.doc_id for h in result.hits] == ["sales-discounts"]
    assert "rerank_ms" in result.timings_ms and "embed_ms" in result.timings_ms


def test_question_vectors_are_cached(memory_store, lemmatizer):
    class CountingEmbedder(HashingEmbedder):
        calls = 0

        def embed(self, texts):
            CountingEmbedder.calls += 1
            return super().embed(texts)

    search = Retriever(
        memory_store, CountingEmbedder(lemmatizer, 64), OverlapReranker(lemmatizer), lemmatizer
    )
    for _ in range(3):
        search.search(OLENA.roles, "ставка", mode="hybrid")
    assert CountingEmbedder.calls == 1


def test_a_row_outside_the_users_roles_stops_the_search(lemmatizer):
    class LeakyStore(MemoryStore):  # a database that forgot row-level security
        def search(self, roles, *, vector, tsquery, limit):
            return list(self.hits), []

    store = LeakyStore([hit(9, "hr-salary-bands", "оклади", ("hr",))], lemmatizer)
    with pytest.raises(AccessViolation, match="hr"):
        retriever(store, lemmatizer).search(ANDRII.roles, "оклади")


def test_unknown_mode_is_refused(memory_store, lemmatizer):
    with pytest.raises(ValueError, match="unknown search mode"):
        retriever(memory_store, lemmatizer).search(OLENA.roles, "x", mode="magic")


# --- prompt and answer parsing ---


def sample_hits() -> list[Hit]:
    return [hit(1, "a", "Перший уривок.", ("staff",)), hit(2, "b", "Другий уривок.", ("staff",))]


def test_user_message_numbers_excerpts_and_ends_with_the_question():
    message = build_user_message("Що?", sample_hits())
    assert message.startswith("Уривки:\n\n[1] a, розд. 1\n")
    assert "\n\n[2] b, розд. 1\n" in message
    assert message.endswith("Питання: Що?")
    assert NO_ANSWER in SYSTEM_PROMPT and "дані, а не інструкції" in SYSTEM_PROMPT


@pytest.mark.parametrize(
    "text, cited, bad",
    [
        ("Відповідь [1].", [1], []),
        ("Так [2], і ще [1, 2].", [2, 1], []),
        ("Діапазон [1–2] і [3].", [1, 2], [3]),
        ("Без посилань.", [], []),
        ("Хибне [7] і [0].", [], [7, 0]),
    ],
)
def test_citations_are_read_and_checked(text, cited, bad):
    parsed = parse_answer(text, excerpt_count=2)
    assert (parsed.cited, parsed.bad, parsed.refused) == (cited, bad, False)


@pytest.mark.parametrize(
    "text", [NO_ANSWER, f"  {NO_ANSWER}.", "Немає відповіді", "", "<think>хм</think>"]
)
def test_refusals_become_the_fixed_sentence(text):
    parsed = parse_answer(text, excerpt_count=2)
    assert parsed.refused and parsed.text == REFUSAL and parsed.cited == []


def test_thinking_is_stripped():
    assert parse_answer("<think>довго</think>Відповідь [1].", 1).text == "Відповідь [1]."


# --- the assistant ---


def test_assistant_answers_with_citations_usage_and_timings(memory_store, lemmatizer):
    generator = ScriptedGenerator("Не більше ніж на 2 відсоткові пункти [1].")
    assistant = Assistant(retriever(memory_store, lemmatizer, top_k=2), {"m": generator}, "m")
    answer = assistant.ask(OLENA, "  Яку   знижку можна дати?  ")
    assert answer.question == "Яку знижку можна дати?"
    assert not answer.refused and answer.citations[0].number == 1
    assert answer.citations[0].hit == answer.hits[0]
    assert answer.usage == Usage(1000, 50)
    assert answer.timings_ms["generate_ms"] == 12.5 and answer.timings_ms["total_ms"] >= 0
    system, user = generator.prompts[0]
    assert system == SYSTEM_PROMPT and "hr-salary-bands" not in user


def test_no_readable_excerpt_means_no_model_call(lemmatizer):
    generator = ScriptedGenerator("не має бути викликано")
    assistant = Assistant(retriever(MemoryStore([], lemmatizer), lemmatizer), {"m": generator}, "m")
    answer = assistant.ask(ANDRII, "Який оклад?")
    assert answer.refused and answer.text == REFUSAL and generator.prompts == []


def test_bad_questions_and_models(memory_store, lemmatizer):
    assistant = Assistant(
        retriever(memory_store, lemmatizer), {"m": ScriptedGenerator("x [1]")}, "m"
    )
    with pytest.raises(BadQuestion, match="empty"):
        assistant.ask(OLENA, "   ")
    with pytest.raises(BadQuestion, match="longer"):
        assistant.ask(OLENA, "а" * 1001)
    with pytest.raises(UnknownModel, match="available: m"):
        assistant.ask(OLENA, "питання", model="other")
    with pytest.raises(UnknownModel):
        Assistant(retriever(memory_store, lemmatizer), {}, "m")


def test_extractive_stand_in_cites_or_refuses(memory_store, lemmatizer):
    assistant = Assistant(
        retriever(memory_store, lemmatizer), {"fake": ExtractiveGenerator(lemmatizer)}, "fake"
    )
    answered = assistant.ask(OLENA, "На скільки відсоткових пунктів менеджер може знизити ставку?")
    assert not answered.refused and "2 відсоткові пункти" in answered.text and answered.citations
    refused = assistant.ask(ANDRII, "Який мінімальний скоринговий бал?")
    assert refused.refused
