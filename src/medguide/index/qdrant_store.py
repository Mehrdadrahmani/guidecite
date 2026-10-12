"""Qdrant vector store: local file mode for development, server mode for deployment."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

PAYLOAD_FIELDS = ["chunk_id", "doc_id", "chunk_index", "source", "title", "url", "section", "text"]


def get_client(cfg: dict):
    from qdrant_client import QdrantClient

    vs = cfg["vector_store"]
    if vs["mode"] == "server":
        return QdrantClient(url=vs["url"])
    local = Path(cfg["paths"]["index_dir"]) / "qdrant"
    local.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(local))


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def build_collection(client, collection: str, records: list[dict], vectors: list[list[float]]):
    from qdrant_client import models

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )
    batch = 256
    for start in range(0, len(records), batch):
        points = [
            models.PointStruct(
                id=point_id(rec["chunk_id"]),
                vector=vec,
                payload={k: rec.get(k) for k in PAYLOAD_FIELDS},
            )
            for rec, vec in zip(
                records[start : start + batch], vectors[start : start + batch], strict=True
            )
        ]
        client.upsert(collection, points=points)


def search(client, collection: str, vector: list[float], k: int) -> list[tuple[dict, float]]:
    res = client.query_points(collection, query=vector, limit=k, with_payload=True)
    return [(dict(p.payload), float(p.score)) for p in res.points]


def manifest_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["index_dir"]) / "manifest.json"


def write_manifest(cfg: dict, info: dict) -> None:
    manifest_path(cfg).write_text(json.dumps(info, indent=2))


def read_manifest(cfg: dict) -> dict:
    path = manifest_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"No index manifest at {path}. Run `make index` first.")
    return json.loads(path.read_text())
