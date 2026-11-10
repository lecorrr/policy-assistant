"""Stand-ins for the embedder, the reranker and the answer model.

They need no GPU, no network and no downloads, and they are deterministic, so tests and
`--fake` runs exercise the whole pipeline. Their quality is beside the point: the hashing embedder
only knows which words two texts share.
"""

import hashlib
import math
import re
import time
from collections.abc import Sequence

from policy_assistant.adapters.chat import Generation
from policy_assistant.answering import NO_ANSWER
from policy_assistant.contracts import Usage
from policy_assistant.text import Lemmatizer


class HashingEmbedder:
    """Hashes each lemma into one of `dims` buckets and normalises the counts."""

    model = "fake-hashing"

    def __init__(self, lemmatizer: Lemmatizer, dims: int = 1024) -> None:
        self.dims = dims
        self._lemmatizer = lemmatizer

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for term in self._lemmatizer.lexemes(text):
            bucket = int(hashlib.sha1(term.encode()).hexdigest()[:8], 16) % self.dims
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:  # pgvector's cosine distance is undefined for a zero vector
            vector[0], norm = 1.0, 1.0
        return [value / norm for value in vector]


class OverlapReranker:
    """Scores a passage by the share of the question's lemmas it contains."""

    name = "fake-overlap"

    def __init__(self, lemmatizer: Lemmatizer) -> None:
        self._lemmatizer = lemmatizer

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        wanted = set(self._lemmatizer.lexemes(query))
        if not wanted:
            return [0.0] * len(passages)
        return [len(wanted & set(self._lemmatizer.lexemes(p))) / len(wanted) for p in passages]


class ExtractiveGenerator:
    """Answers with the excerpt sentence that shares most lemmas with the question, cited.

    Reads the excerpts back out of the prompt answering.build_user_message wrote. Refuses when no
    sentence shares at least half of the question's lemmas.
    """

    def __init__(self, lemmatizer: Lemmatizer, name: str = "fake") -> None:
        self.name = name
        self._lemmatizer = lemmatizer

    def generate(self, system: str, user: str) -> Generation:
        started = time.perf_counter()
        question = user.rsplit("Питання:", 1)[-1].strip()
        wanted = set(self._lemmatizer.lexemes(question))
        best = (0.0, "", 0)
        for number, body in _excerpts(user):
            for sentence in re.split(r"(?<=[.;:])\s+|\n", body):
                shared = len(wanted & set(self._lemmatizer.lexemes(sentence)))
                score = shared / max(len(wanted), 1)
                if sentence.strip() and score > best[0]:
                    best = (score, sentence.strip(), number)
        score, sentence, number = best
        text = f"{sentence} [{number}]" if score >= 0.5 else NO_ANSWER
        return Generation(
            text=text,
            usage=Usage(input_tokens=len(system + user) // 4, output_tokens=len(text) // 4),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )


def _excerpts(user: str) -> list[tuple[int, str]]:
    found = []
    body = user.split("Уривки:", 1)[-1].rsplit("Питання:", 1)[0]
    for block in re.split(r"\n\n(?=\[\d+\] )", body.strip()):
        match = re.match(r"\[(\d+)\] [^\n]*\n[^\n]*\n(.*)", block, re.DOTALL)
        if match:
            found.append((int(match.group(1)), match.group(2)))
    return found
