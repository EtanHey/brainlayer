"""Vector index for style message embeddings (ChromaDB)."""

import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from .unified_timeline import UnifiedMessage

DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "zikaron" / "chromadb"
STYLE_COLLECTION = "style_messages"
CHROMADB_BATCH_SIZE = 1000


def _msg_id(msg: UnifiedMessage, idx: int) -> str:
    """Deterministic ID for a message."""
    content = f"{msg.timestamp.isoformat()}|{msg.source}|{msg.text[:100]}"
    h = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"style_{idx}_{h}"


def _timestamp_epoch(msg: UnifiedMessage) -> float:
    """Unix epoch for ChromaDB numeric filter."""
    return msg.timestamp.timestamp()


def get_style_client(db_path: Path = DEFAULT_DB_PATH) -> chromadb.Client:
    """Get ChromaDB client for style index."""
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def get_style_collection(client: chromadb.Client) -> chromadb.Collection:
    """Get or create the style messages collection."""
    return client.get_or_create_collection(
        name=STYLE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def index_style_messages(
    messages_with_embeddings: list[tuple[UnifiedMessage, list[float]]],
    collection: chromadb.Collection,
) -> int:
    """Index style messages with embeddings. Replaces existing data."""
    if not messages_with_embeddings:
        return 0

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, (msg, emb) in enumerate(messages_with_embeddings):
        ids.append(_msg_id(msg, i))
        embeddings.append(emb)
        documents.append(msg.text[:2000])
        metadatas.append({
            "ts_epoch": _timestamp_epoch(msg),
            "timestamp": msg.timestamp.isoformat(),
            "source": msg.source,
            "language": msg.language,
            "chat_id": (msg.chat_id or "")[:200],
            "contact_name": (msg.contact_name or "")[:200],
            "relationship_tag": msg.relationship_tag or "unlabeled",
        })

    for i in range(0, len(ids), CHROMADB_BATCH_SIZE):
        end = min(i + CHROMADB_BATCH_SIZE, len(ids))
        collection.upsert(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )

    return len(ids)


def clear_style_collection(collection: chromadb.Collection) -> None:
    """Clear all documents from the style collection."""
    # ChromaDB delete requires IDs; get all and delete
    result = collection.get(include=[])
    if result["ids"]:
        collection.delete(ids=result["ids"])


def query_style_messages(
    collection: chromadb.Collection,
    query_embeddings: list[list[float]],
    n_results: int = 100,
    where: Optional[dict] = None,
) -> dict:
    """Query style messages by embedding. Returns documents and metadatas."""
    return collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )


def get_embeddings_for_batch(
    collection: chromadb.Collection,
    start_epoch: float,
    end_epoch: float,
    language: Optional[str] = None,
) -> tuple[list[list[float]], list[str]]:
    """
    Get all embeddings and documents for a time range (and optionally language).

    Returns:
        (embeddings, documents) for use in cluster sampling.
    """
    where: dict = {
        "$and": [
            {"ts_epoch": {"$gte": start_epoch}},
            {"ts_epoch": {"$lte": end_epoch}},
        ]
    }
    if language and language != "all":
        where["$and"].append({"language": language})

    # Get all - no query_embeddings, use get with where
    result = collection.get(
        where=where,
        limit=10000,
        include=["embeddings", "documents"],
    )
    embs = result.get("embeddings") or []
    docs = result.get("documents") or []
    return (embs, docs)
