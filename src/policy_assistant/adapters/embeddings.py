from collections.abc import Sequence

import httpx


class EmbeddingError(RuntimeError):
    pass


class OllamaEmbedder:
    def __init__(
        self, base_url: str, model: str, dims: int, *, timeout_s: float = 120, batch_size: int = 16
    ) -> None:
        self.model = model
        self.dims = dims
        self._base_url = base_url.rstrip("/")
        self._batch_size = batch_size
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout_s)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(self._embed_batch(batch))
        return vectors

    def close(self) -> None:
        self._client.close()

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = self._client.post("/api/embed", json={"model": self.model, "input": batch})
        except httpx.TransportError as error:
            raise EmbeddingError(
                f"Ollama is not reachable at {self._base_url} ({type(error).__name__}): "
                "is it running? `ollama serve` starts it"
            ) from None
        if response.status_code == 404:
            raise EmbeddingError(
                f"Ollama has no model {self.model!r}: run `ollama pull {self.model}`"
            )
        if response.status_code != 200:
            raise EmbeddingError(f"Ollama answered {response.status_code}: {response.text[:200]}")
        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(batch):
            raise EmbeddingError(
                f"Ollama returned {len(embeddings or [])} embeddings for {len(batch)} texts"
            )
        for vector in embeddings:
            if len(vector) != self.dims:
                raise EmbeddingError(
                    f"{self.model} makes {len(vector)}-dimension embeddings, but "
                    f"POLICY_EMBEDDING_DIMS is {self.dims}: set it to {len(vector)} and run "
                    "`init-db --reset`"
                )
        return embeddings
