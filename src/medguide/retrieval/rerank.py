"""Cross-encoder reranking."""

from __future__ import annotations


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device, max_length=512)
        self.batch_size = batch_size

    def rerank(self, query: str, records: list[dict], top_k: int) -> list[tuple[dict, float]]:
        if not records:
            return []
        scores = self.model.predict(
            [(query, r["text"]) for r in records],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        ranked = sorted(zip(records, scores, strict=True), key=lambda x: -float(x[1]))
        return [(r, float(s)) for r, s in ranked[:top_k]]
