from collections.abc import Sequence
from pathlib import Path


class RerankerError(RuntimeError):
    pass


class CrossEncoderReranker:
    """CPU cross-encoder; the model (~0.6 GB) downloads from Hugging Face on first use."""

    def __init__(self, model_name: str, cache_dir: Path, threads: int | None = None) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._encoder = TextCrossEncoder(
                model_name=model_name, cache_dir=str(cache_dir), threads=threads
            )
        except ValueError as error:  # fastembed's answer to an unknown model name
            raise RerankerError(f"fastembed cannot load {model_name!r}: {error}") from None
        except OSError as error:
            raise RerankerError(
                f"cannot download or open {model_name!r} in {cache_dir}: {error}. The first run "
                "needs internet access to huggingface.co"
            ) from None

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        try:
            return [
                float(score) for score in self._encoder.rerank(query, list(passages), batch_size=8)
            ]
        except Exception as error:  # onnxruntime raises its own exception types
            raise RerankerError(f"the reranker failed: {type(error).__name__}: {error}") from error


class NoReranker:
    """Keeps the fused order (no reranking)."""

    name = "none"

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        return [float(-position) for position in range(len(passages))]


def build_reranker(name: str, cache_dir: Path, threads: int | None = None):
    if name == "none":
        return NoReranker()
    return CrossEncoderReranker(name, cache_dir, threads)
