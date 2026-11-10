"""Retrieval quality per search mode, and the retrieval leak test."""

from collections.abc import Callable

from policy_assistant.config import UsersConfig
from policy_assistant.evaluation.cases import EvalSet
from policy_assistant.search import MODES, AccessViolation, Retriever

K = 5


def evaluate_retrieval(
    eval_set: EvalSet,
    retriever: Retriever,
    users: UsersConfig,
    modes: tuple[str, ...] = MODES,
    *,
    progress: Callable[[str], None] = lambda line: None,
) -> dict:
    """recall@5 is the share of answerable questions with their evidence in the top 5 excerpts;
    MRR averages 1 / rank of the first such excerpt (0 when it is missing)."""
    per_question = []
    for question in eval_set.answerable:
        user = users.user(question.ask_as)
        row: dict = {"id": question.id, "ask_as": question.ask_as, "ranks": {}, "top": {}}
        for mode in modes:
            result = retriever.search(user.roles, question.question, mode=mode, top_k=K)
            rank = next(
                (
                    position
                    for position, hit in enumerate(result.hits, start=1)
                    if any(source.matches(hit) for source in question.sources)
                ),
                None,
            )
            row["ranks"][mode] = rank
            row["top"][mode] = [hit.label for hit in result.hits]
        per_question.append(row)
        ranks = "  ".join(f"{mode} {row['ranks'][mode] or '-'}" for mode in modes)
        progress(f"{question.id}  {ranks}")

    summary = {}
    for mode in modes:
        ranks = [row["ranks"][mode] for row in per_question]
        found = [rank for rank in ranks if rank is not None]
        summary[mode] = {
            "recall_at_5": round(len(found) / len(ranks), 3),
            "mrr": round(sum(1 / rank for rank in found) / len(ranks), 3),
            "missed": [row["id"] for row in per_question if row["ranks"][mode] is None],
        }
    items = len(eval_set.questions) + len(eval_set.injections)
    progress(f"Leak test: {items} questions as each of {len(users.users)} users...")
    return {
        "k": K,
        "modes": summary,
        "questions": per_question,
        "leaks": leak_test(eval_set, retriever, users),
    }


def leak_test(eval_set: EvalSet, retriever: Retriever, users: UsersConfig) -> dict:
    """Every question and injection as every user, checking all rows the database returns.

    Fails a check when a row's roles do not cover the user, or when a leak user gets any chunk
    of the question's sources.
    """
    checks = 0
    failures = []
    items = [*eval_set.questions, *eval_set.injections]
    for login in sorted(users.users):
        user = users.user(login)
        for item in items:
            checks += 1
            try:
                result = retriever.search(user.roles, item.question, mode="hybrid", top_k=K)
            except AccessViolation as error:
                failures.append({"id": item.id, "user": login, "problem": str(error)})
                continue
            if login in getattr(item, "leak_as", []):
                leaked = [
                    hit.label
                    for hit in result.candidates
                    if any(source.matches(hit) for source in item.sources)
                ]
                if leaked:
                    failures.append({"id": item.id, "user": login, "problem": f"got {leaked}"})
    return {"checks": checks, "failures": failures, "passed": not failures}
