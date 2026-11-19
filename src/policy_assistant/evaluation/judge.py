import json
import re

from pydantic import BaseModel, ValidationError

from policy_assistant.adapters.chat import ModelError
from policy_assistant.contracts import Hit
from policy_assistant.evaluation.cases import without_citations

JUDGE_SYSTEM = "\n".join(
    [
        "You check answers from a document assistant for faithfulness.",
        "You get the excerpts the assistant was given, the question and the assistant's answer, "
        "all in Ukrainian.",
        "Split the answer into its factual claims, ignoring citation markers such as [2]. For each "
        "claim decide whether the excerpts support it: a claim is supported only if an excerpt "
        "states it or it follows directly from one. What you know from elsewhere does not count.",
        'Reply with JSON only, in exactly this shape: {"claims": [{"claim": "...", '
        '"supported": true}]}',
    ]
)


class _Claim(BaseModel):
    claim: str
    supported: bool


class _Verdict(BaseModel):
    claims: list[_Claim]


class JudgeError(RuntimeError):
    pass


def judge_message(question: str, answer: str, hits: list[Hit]) -> str:
    excerpts = "\n\n".join(f"[{n}] {hit.label}\n{hit.body}" for n, hit in enumerate(hits, start=1))
    return f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}\n\nAnswer: {without_citations(answer)}"


def parse_verdict(text: str) -> float | None:
    """Supported claims / all claims, or None for an answer with no claims."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise JudgeError(f"the judge did not reply with JSON: {text[:120]!r}")
    try:
        verdict = _Verdict.model_validate(json.loads(match.group()))
    except (json.JSONDecodeError, ValidationError) as error:
        raise JudgeError(f"the judge's JSON has the wrong shape: {error}") from None
    if not verdict.claims:
        return None
    return sum(claim.supported for claim in verdict.claims) / len(verdict.claims)


def judge_faithfulness(
    judge, question: str, answer: str, hits: list[Hit]
) -> tuple[float | None, object]:
    """Return (score, generation) so the caller can count the judge's tokens."""
    generation = judge.generate(JUDGE_SYSTEM, judge_message(question, answer, hits))
    return parse_verdict(generation.text), generation


_CHECK_EXCERPT = Hit(
    chunk_id=0,
    doc_id="check",
    section="1",
    label="Перевірка судді",
    heading="Перевірка судді",
    body="Щорічна оплачувана відпустка триває 14 календарних днів.",
    roles=("staff",),
    url=None,
)


def check_judge(judge) -> None:
    """One cheap verdict before a run, so a judge without credit fails fast."""
    try:
        judge_faithfulness(
            judge, "Скільки триває відпустка?", "14 календарних днів [1].", [_CHECK_EXCERPT]
        )
    except (JudgeError, ModelError) as error:
        raise JudgeError(f"the judge {judge.name} does not work: {error}") from error
