"""Sentence-transformer embeddings exposed through LangChain's Embeddings interface."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings


class GuideEmbedder(Embeddings):
    """bge-style embedder: query instruction prefix on queries only, L2-normalised outputs."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        query_instruction: str = "",
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.normalize = normalize

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, show_progress=len(texts) > 256)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_instruction + text])[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.query_instruction + t for t in texts])


def load_embedder(cfg: dict, model_name: str | None = None) -> GuideEmbedder:
    from medguide.config import embedding_settings

    default_model, device, batch = embedding_settings(cfg)
    emb = cfg["embedding"]
    return GuideEmbedder(
        model_name=model_name or default_model,
        device=device,
        batch_size=batch,
        query_instruction=emb.get("query_instruction", ""),
        normalize=emb.get("normalize", True),
    )
