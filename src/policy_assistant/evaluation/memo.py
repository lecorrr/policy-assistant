"""The one-page model-choice memo, rendered from the evaluation reports."""

from policy_assistant.config import ModelsConfig

# One question is about two points of accuracy; models this many questions apart are a tie.
NOISE_QUESTIONS = 2
# Below this cost per 1,000 questions the price hardly matters, so speed decides.
EVERYDAY_COST_LIMIT_USD = 1.0


def render_memo(
    answer_reports: dict[str, dict], retrieval: dict | None, models: ModelsConfig
) -> str:
    """answer_reports maps a model name to its latest report, in the memo's order."""
    names = [name for name in models.memo if name in answer_reports]
    if not names:
        raise ValueError("no answer reports for the memo's models: run `eval-answers` first")
    first = answer_reports[names[0]]
    summaries = {name: answer_reports[name]["summary"] for name in names}

    lines = [
        "# Which model should answer staff questions?",
        "",
        f"Measured {first['finished_at'][:10]} with policy-assistant {first['version']}: "
        f"{summaries[names[0]]['answerable']} questions in Ukrainian with an answer in the "
        f"documents, {summaries[names[0]]['unanswerable']} without one, and "
        f"{summaries[names[0]]['leak_cases']} attempts to read documents outside the asker's "
        "role. "
        f"Every model got the same excerpts from the same search: {_search_setup(first)}. "
        f"The questions were written against the law's edition of {first['law_edition']}. "
        f"Prices are list prices checked {models.prices_checked:%d.%m.%Y}.",
        "",
        "| | " + " | ".join(_model_title(name, models) for name in names) + " |",
        "|---|" + "---|" * len(names),
    ]
    rows = [
        ("Correct answers", lambda s: f"{s['correct']} / {s['answerable']}"),
        (
            "Faithful to the excerpts",
            _faithfulness_cell,
        ),
        ("Numbers found in the excerpts", lambda s: _pct(s["numbers_grounded"])),
        ("Cites the right source", lambda s: _pct(s["cites_source"])),
        (
            "Refused with no answer in the documents",
            lambda s: f"{s['refused_unanswerable']} / {s['unanswerable']}",
        ),
        ("Refused although the answer was there", lambda s: str(s["false_refusals"])),
        ("Leaks", lambda s: f"{s['leaks']} / {s['leak_cases']}"),
        ("p95 latency, question to answer", lambda s: _seconds(s["latency_ms"]["p95"])),
        ("Cost per 1,000 questions", None),
    ]
    for title, cell in rows:
        if cell is None:
            values = [_cost(name, summaries[name], models) for name in names]
        else:
            values = [cell(summaries[name]) for name in names]
        lines.append(f"| {title} | " + " | ".join(values) + " |")

    if retrieval:
        modes = retrieval["modes"]
        lines += [
            "",
            "Search is shared by all models. recall@5 (evidence in the five excerpts): "
            + ", ".join(f"{mode} {modes[mode]['recall_at_5']:.2f}" for mode in modes)
            + f". Retrieval leak test: {retrieval['leaks']['checks']} checks, "
            + (
                "none failed."
                if retrieval["leaks"]["passed"]
                else f"{len(retrieval['leaks']['failures'])} FAILED."
            ),
        ]

    search_misses = set()
    if retrieval:
        search_misses = set(retrieval["modes"].get("hybrid+rerank", {}).get("missed", []))
    wrong = {name: _wrong_answers(answer_reports[name]) for name in names}
    lines += [
        "",
        "## Which model for what",
        "",
        *_recommendations(names, summaries, models, wrong, search_misses),
    ]
    judge = first.get("judge")
    lines += [
        "",
        "## How to read this",
        "",
        "- One question is about two points of accuracy, so models within "
        f"{NOISE_QUESTIONS} questions of each other are a tie. Everyday questions go to the "
        "fastest of the models tied with the best, among those under "
        f"${EVERYDAY_COST_LIMIT_USD:.0f} per 1,000 questions.",
        "- A leak counts when a user got an excerpt they may not read, or an answer matching a "
        "restricted fact. Row-level security filters search, so a leak would mean a broken setup "
        "rather than a weak model.",
        _faithfulness_note(judge),
        "- Latency is measured on this machine; cloud models include the network round trip.",
        "- Every answer and score behind these numbers is in the evals/ folder next to this memo.",
    ]
    return "\n".join(lines) + "\n"


def _wrong_answers(report: dict) -> set[str]:
    """Answerable questions the model got wrong, refusals included."""
    return {
        case["id"]
        for case in report.get("cases", [])
        if case["kind"] == "question" and "answered_answerable" in case and not case["correct"]
    }


def _recommendations(
    names: list[str],
    summaries: dict[str, dict],
    models: ModelsConfig,
    wrong: dict[str, set[str]],
    search_misses: set[str],
) -> list[str]:
    safe = [name for name in names if summaries[name]["leaks"] == 0]
    if not safe:
        return ["- No model passed the leak test. Fix the setup before choosing a model."]

    def right(name: str) -> int:
        return summaries[name]["correct"]

    def p95(name: str) -> float:
        return summaries[name]["latency_ms"]["p95"] or float("inf")

    def cost(name: str) -> float:
        return _cost_value(name, summaries[name], models)

    most_right = max(right(name) for name in safe)
    tied = [name for name in safe if right(name) >= most_right - NOISE_QUESTIONS]
    affordable = [name for name in tied if cost(name) <= EVERYDAY_COST_LIMIT_USD]
    if affordable:
        everyday = min(affordable, key=lambda name: (p95(name), name))
    else:
        everyday = min(tied, key=lambda name: (cost(name), name))
    cheapest = min(tied, key=lambda name: (cost(name), name))
    lines = [f"- Everyday questions: **{everyday}**, {_profile(everyday, summaries, models)}."]
    if cheapest != everyday:
        lines.append(
            f"- Lowest running cost: **{cheapest}**, {_profile(cheapest, summaries, models)}."
        )

    ranked = sorted(safe, key=right, reverse=True)
    if len(ranked) == 1 or right(ranked[0]) - right(ranked[1]) > NOISE_QUESTIONS:
        ahead = "the only model measured"
        if len(ranked) > 1:
            ahead = f"{right(ranked[0]) - right(ranked[1])} more right than the next"
        lines.append(
            f"- Questions where a wrong answer is costly, such as the law: **{ranked[0]}**, "
            f"{ahead}."
        )
    else:
        if len({right(name) for name in tied}) == 1:
            how_close = f"{', '.join(tied)} each got {most_right} right"
        else:
            how_close = f"{', '.join(tied)} are within {NOISE_QUESTIONS} questions of the best"
        lines.append(
            "- Questions where a wrong answer is costly, such as the law: no model is more "
            f"accurate here; {how_close}."
        )

    missed = sorted(set().union(*(wrong[name] for name in safe)) & search_misses)
    if missed:
        everyone = all(wrong[name] <= search_misses for name in safe)
        lines.append(
            f"- {', '.join(missed)}: search did not find the evidence, so no model could answer "
            + ("it; better search, not another model, is what would help." if everyone else "it.")
        )

    local = [name for name in safe if models.models[name].provider == "ollama"]
    for name in local:
        # Decided on the whole points the memo prints, so the verdict matches the number shown.
        best_accuracy = max(summaries[other]["accuracy"] for other in safe)
        gap = round(100 * (best_accuracy - summaries[name]["accuracy"]))
        if gap <= 10:
            verdict = (
                "good enough to keep HR and risk questions on our own machine "
                f"(p95 {_seconds(summaries[name]['latency_ms']['p95'])})"
            )
        else:
            verdict = "too far behind to rely on: sensitive questions need a contracted provider"
        lines.append(
            f"- Data that must not leave the company: **{name}** is {gap} points behind the best, "
            f"{verdict}."
        )
    return lines


def _profile(name: str, summaries: dict[str, dict], models: ModelsConfig) -> str:
    summary = summaries[name]
    cost = _cost(name, summary, models)
    if cost.startswith("$"):
        cost += " per 1,000 questions"
    return (
        f"{summary['correct']} / {summary['answerable']} right, p95 "
        f"{_seconds(summary['latency_ms']['p95'])}, {cost}"
    )


def _search_setup(report: dict) -> str:
    setup = report.get("retrieval")
    if not setup:
        return "full text and vectors, fused and reranked"
    reranked = "not reranked" if setup["reranker"] == "none" else f"reranked by {setup['reranker']}"
    return (
        f"{setup['embedding_model']} vectors and full text, fused, {reranked}, "
        f"top {setup['top_k']} to the model"
    )


def _faithfulness_cell(summary: dict) -> str:
    # Reports from before 0.1.1 carry no judge counts, so their share is shown as it is.
    answered, errors = summary.get("answered"), summary.get("judge_errors", 0)
    if summary["faithfulness"] is None:
        return f"judge failed on all {answered}" if errors else "not judged"
    share = _pct(summary["faithfulness"])
    if answered is not None and summary["judged"] < answered:
        return f"{share} ({summary['judged']} of {answered} judged)"
    return share


def _faithfulness_note(judge: str | None) -> str:
    if judge:
        return f"- Faithfulness is the share of claims the excerpts support, judged by {judge}."
    return (
        "- Faithfulness was not judged; the numbers row checks every figure against the excerpts."
    )


def _model_title(name: str, models: ModelsConfig) -> str:
    return f"{name} ({models.models[name].model})" if models.models[name].model != name else name


def _cost(name: str, summary: dict, models: ModelsConfig) -> str:
    if models.models[name].provider == "ollama":
        return "no token cost (own GPU)"
    value = summary["cost_per_1000_questions_usd"]
    return f"${value:.2f}" if value is not None else "n/a"


def _cost_value(name: str, summary: dict, models: ModelsConfig) -> float:
    if models.models[name].provider == "ollama":
        return 0.0
    return summary["cost_per_1000_questions_usd"] or 0.0


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{round(100 * value)} %"


def _seconds(ms: float | None) -> str:
    return "n/a" if ms is None else f"{ms / 1000:.1f} s"
