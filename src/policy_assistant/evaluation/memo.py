"""The one-page model-choice memo, rendered from the evaluation reports."""

from policy_assistant.config import ModelsConfig


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
        "The law is the edition of "
        f"{first['law_edition']}. Prices are list prices checked {models.prices_checked:%d.%m.%Y}.",
        "",
        "| | " + " | ".join(_model_title(name, models) for name in names) + " |",
        "|---|" + "---|" * len(names),
    ]
    rows = [
        ("Correct answers", lambda s: f"{s['correct']} / {s['answerable']}"),
        (
            "Faithful to the excerpts",
            lambda s: _pct(s["faithfulness"]) if s["faithfulness"] is not None else "not judged",
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

    lines += ["", "## Which model for what", "", *_recommendations(names, summaries, models)]
    judge = first.get("judge")
    lines += [
        "",
        "## How to read this",
        "",
        "- One question is about two points of accuracy, so a difference of a question or two is "
        "noise.",
        "- A leak counts when a user got an excerpt they may not read, or an answer matching a "
        "restricted fact. Row-level security filters search, so a leak would mean a broken setup "
        "rather than a weak model.",
        _faithfulness_note(judge),
        "- Latency is measured on this machine; cloud models include the network round trip.",
        "- The cases behind every number are in data/evals/.",
    ]
    return "\n".join(lines) + "\n"


def _recommendations(
    names: list[str], summaries: dict[str, dict], models: ModelsConfig
) -> list[str]:
    safe = [name for name in names if summaries[name]["leaks"] == 0]
    if not safe:
        return ["- No model passed the leak test. Fix the setup before choosing a model."]
    best = max(summaries[name]["accuracy"] for name in safe)
    close = [name for name in safe if summaries[name]["accuracy"] >= best - 0.05]
    cheapest = min(close, key=lambda name: (_cost_value(name, summaries[name], models), name))
    most_accurate = max(
        safe, key=lambda name: (summaries[name]["accuracy"], summaries[name]["faithfulness"] or 0)
    )
    fastest = min(safe, key=lambda name: summaries[name]["latency_ms"]["p95"] or float("inf"))
    lines = [
        f"- Everyday questions: **{cheapest}**, the cheapest model within five points of the best "
        f"accuracy ({_pct(summaries[cheapest]['accuracy'])} against {_pct(best)}).",
        f"- Questions where a wrong answer is costly, such as the law: **{most_accurate}**, the "
        f"most accurate ({_pct(summaries[most_accurate]['accuracy'])}).",
        f"- When speed matters most: **{fastest}**, p95 "
        f"{_seconds(summaries[fastest]['latency_ms']['p95'])}.",
    ]
    local = [name for name in safe if models.models[name].provider == "ollama"]
    for name in local:
        # Decided on the whole points the memo prints, so the verdict matches the number shown.
        gap = round(100 * (best - summaries[name]["accuracy"]))
        if gap <= 10:
            verdict = "good enough to keep HR and risk questions on our own machine"
        else:
            verdict = "too far behind to rely on: sensitive questions need a contracted provider"
        lines.append(
            f"- Data that must not leave the company: **{name}** is {gap} points behind the best, "
            f"{verdict}."
        )
    return lines


def _search_setup(report: dict) -> str:
    setup = report.get("retrieval")
    if not setup:
        return "full text and vectors, fused and reranked"
    reranked = "not reranked" if setup["reranker"] == "none" else f"reranked by {setup['reranker']}"
    return (
        f"{setup['embedding_model']} vectors and full text, fused, {reranked}, "
        f"top {setup['top_k']} to the model"
    )


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
