import time
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from policy_assistant.contracts import Hit
from policy_assistant.ranking import reciprocal_rank_fusion
from policy_assistant.text import Lemmatizer, tsquery_literal

MODES = ("vector", "lexical", "hybrid", "hybrid+rerank")


class AccessViolation(RuntimeError):
    """A row the user's roles don't cover; row-level security should make this impossible."""


class Store(Protocol):
    def search(
        self,
        roles: Iterable[str],
        *,
        vector: Sequence[float] | None,
        tsquery: str | None,
        limit: int,
    ) -> tuple[list[Hit], list[Hit]]: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    name: str

    def scores(self, query: str, passages: Sequence[str]) -> list[float]: ...


@dataclass
class SearchResult:
    hits: list[Hit]  # best first, at most top_k
    candidates: list[Hit]  # every row the database returned, for leak checks
    timings_ms: dict[str, float] = field(default_factory=dict)


class Retriever:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        reranker: Reranker,
        lemmatizer: Lemmatizer,
        *,
        candidates: int = 30,
        rerank_top: int = 20,
        top_k: int = 5,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.lemmatizer = lemmatizer
        self.candidates = candidates
        self.rerank_top = rerank_top
        self.top_k = top_k
        self._vectors: OrderedDict[str, list[float]] = OrderedDict()

    def search(
        self,
        roles: Sequence[str],
        question: str,
        *,
        mode: str = "hybrid+rerank",
        top_k: int | None = None,
    ) -> SearchResult:
        if mode not in MODES:
            raise ValueError(f"unknown search mode {mode!r}; choose from {', '.join(MODES)}")
        top_k = top_k or self.top_k
        timings: dict[str, float] = {}

        vector = None
        if mode != "lexical":
            started = time.perf_counter()
            vector = self._vector(question)
            timings["embed_ms"] = _since(started)
        tsquery = tsquery_literal(self.lemmatizer.lexemes(question)) if mode != "vector" else None

        started = time.perf_counter()
        vector_hits, lexical_hits = self.store.search(
            roles, vector=vector, tsquery=tsquery, limit=self.candidates
        )
        timings["search_ms"] = _since(started)
        candidates = _unique([*vector_hits, *lexical_hits])
        _check_access(roles, candidates)

        if mode == "vector":
            ranked = vector_hits
        elif mode == "lexical":
            ranked = lexical_hits
        else:
            ranked = self._fuse(vector_hits, lexical_hits)
            if mode == "hybrid+rerank":
                started = time.perf_counter()
                ranked = self._rerank(question, ranked[: self.rerank_top])
                timings["rerank_ms"] = _since(started)
        return SearchResult(hits=ranked[:top_k], candidates=candidates, timings_ms=timings)

    def _vector(self, question: str) -> list[float]:
        vector = self._vectors.get(question)
        if vector is None:
            vector = self.embedder.embed([question])[0]
            self._vectors[question] = vector
            if len(self._vectors) > 512:
                self._vectors.popitem(last=False)
        return vector

    def _fuse(self, vector_hits: list[Hit], lexical_hits: list[Hit]) -> list[Hit]:
        by_id = {hit.chunk_id: hit for hit in [*lexical_hits, *vector_hits]}
        fused = reciprocal_rank_fusion(
            [[hit.chunk_id for hit in vector_hits], [hit.chunk_id for hit in lexical_hits]]
        )
        return [replace(by_id[chunk_id], score=score) for chunk_id, score in fused]

    def _rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        scores = self.reranker.scores(question, [hit.search_text for hit in hits])
        rescored = [replace(hit, score=score) for hit, score in zip(hits, scores, strict=True)]
        return sorted(rescored, key=lambda hit: -hit.score)


def _check_access(roles: Sequence[str], hits: list[Hit]) -> None:
    allowed = set(roles)
    for hit in hits:
        if not allowed.intersection(hit.roles):
            raise AccessViolation(
                f"chunk {hit.chunk_id} ({hit.label}) is for {', '.join(hit.roles)}, not for "
                f"{', '.join(sorted(allowed)) or 'no roles'}: is the app connected as the reader?"
            )


def _unique(hits: list[Hit]) -> list[Hit]:
    seen: dict[int, Hit] = {}
    for hit in hits:
        seen.setdefault(hit.chunk_id, hit)
    return list(seen.values())


def _since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
