"""Combining ranked lists."""

from collections.abc import Sequence

# The usual constant from Cormack et al. (2009): it damps the advantage of the very top ranks.
RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = RRF_K
) -> list[tuple[int, float]]:
    """Fuses ranked lists of ids into one list of (id, score), best first.

    Each id scores the sum of 1 / (k + rank) over the lists it appears in, ranks counted from 1.
    Ties keep the order in which ids were first seen, so the result is deterministic.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    order = {item: position for position, item in enumerate(scores)}
    return sorted(scores.items(), key=lambda pair: (-pair[1], order[pair[0]]))
