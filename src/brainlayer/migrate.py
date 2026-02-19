"""Migration script to convert ChromaDB data to sqlite-vec."""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from .vector_store import VectorStore
from .embeddings import get_embedding_model

logger = logging.getLogger(__name__)

# Paths
CHROMADB_PATH = Path.home() / ".local" / "share" / "brainlayer" / "chromadb.backup"
SQLITE_PATH = Path.home() / ".local" / "share" / "brainlayer" / "brainlayer.db"


def migrate_from_chromadb() -> bool:
    """Migrate data from ChromaDB to sqlite-vec."""
    if not CHROMADB_AVAILABLE:
        print("ChromaDB not available, skipping migration")
        return False
    
    if not CHROMADB_PATH.exists():
        print("No existing ChromaDB found, skipping migration")
        return False
    
    print(f"Migrating from ChromaDB at {CHROMADB_PATH}")
    
    try:
        # Connect to ChromaDB
        client = chromadb.PersistentClient(
            path=str(CHROMADB_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get all collections
        collections = client.list_collections()
        if not collections:
            print("No collections found in ChromaDB")
            return False

        collection_names = [c.name for c in collections]
        total_all = sum(c.count() for c in collections)
        print(f"Found {len(collections)} collections with {total_all} total chunks: {collection_names}")

        # Create sqlite-vec store
        print(f"Creating sqlite-vec database at {SQLITE_PATH}")
        vector_store = VectorStore(SQLITE_PATH)

        # Lazy-load embedding model only if needed
        embedding_model = None
        grand_total = 0

        for collection in collections:
            total_count = collection.count()
            print(f"\n--- Migrating collection '{collection.name}' ({total_count} chunks) ---")

            # Fetch and insert in batches to avoid OOM
            FETCH_BATCH = 5000
            migrated = 0
            need_reembed_total = 0

            for offset in range(0, total_count, FETCH_BATCH):
                batch_data = collection.get(
                    include=["documents", "metadatas", "embeddings"],
                    limit=FETCH_BATCH,
                    offset=offset,
                )

                if not batch_data["ids"]:
                    break

                chunks = []
                embeddings_batch = []
                need_reembed = []

                for i, chunk_id in enumerate(batch_data["ids"]):
                    document = batch_data["documents"][i]
                    metadata = batch_data["metadatas"][i] or {}
                    embedding = batch_data["embeddings"][i] if batch_data["embeddings"] else None

                    chunk_data = {
                        "id": chunk_id,
                        "content": document,
                        "metadata": {k: v for k, v in metadata.items()
                                     if k not in ["source_file", "project", "content_type", "value_type", "char_count"]},
                        "source_file": metadata.get("source_file", "unknown"),
                        "project": metadata.get("project"),
                        "content_type": metadata.get("content_type"),
                        "value_type": metadata.get("value_type"),
                        "char_count": metadata.get("char_count", len(document))
                    }
                    chunks.append(chunk_data)

                    if embedding and len(embedding) == 1024:
                        embeddings_batch.append(embedding)
                    else:
                        embeddings_batch.append(None)
                        need_reembed.append(i)

                # Re-embed chunks with wrong/missing dimensions
                if need_reembed:
                    need_reembed_total += len(need_reembed)
                    if embedding_model is None:
                        print("Loading bge-large-en-v1.5 for re-embedding...")
                        embedding_model = get_embedding_model()

                    for rb_start in range(0, len(need_reembed), 32):
                        rb_indices = need_reembed[rb_start:rb_start + 32]
                        rb_texts = [chunks[idx]["content"] for idx in rb_indices]
                        try:
                            rb_embs = embedding_model._load_model().encode(
                                rb_texts, convert_to_numpy=True, show_progress_bar=False
                            )
                            for j, idx in enumerate(rb_indices):
                                embeddings_batch[idx] = rb_embs[j].tolist()
                        except Exception as e:
                            logger.error(f"Failed to re-embed batch: {e}")
                            for idx in rb_indices:
                                embeddings_batch[idx] = [0.0] * 1024

                # Insert this batch into sqlite-vec
                INSERT_BATCH = 1000
                for ins_start in range(0, len(chunks), INSERT_BATCH):
                    ins_chunks = chunks[ins_start:ins_start + INSERT_BATCH]
                    ins_embs = embeddings_batch[ins_start:ins_start + INSERT_BATCH]
                    vector_store.upsert_chunks(ins_chunks, ins_embs)

                migrated += len(chunks)
                reembed_note = f" ({len(need_reembed)} re-embedded)" if need_reembed else ""
                print(f"  {migrated}/{total_count} chunks{reembed_note}")

            grand_total += migrated
            if need_reembed_total:
                print(f"  ({need_reembed_total} total re-embedded in this collection)")

        vector_store.close()

        # Verify migration
        vector_store = VectorStore(SQLITE_PATH)
        final_count = vector_store.count()
        vector_store.close()

        print(f"\nMigration complete: {final_count} chunks in sqlite-vec (from {total_all} in ChromaDB)")

        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        print(f"Migration failed: {e}")
        return False


def main():
    """Main migration entry point."""
    logging.basicConfig(level=logging.INFO)
    
    print("זיכרון - Migration Tool")
    print("=" * 50)
    
    if SQLITE_PATH.exists():
        response = input(f"sqlite-vec database already exists at {SQLITE_PATH}. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Migration cancelled")
            return
        
        # Remove existing database and WAL/SHM files
        SQLITE_PATH.unlink()
        for suffix in ["-shm", "-wal"]:
            p = SQLITE_PATH.parent / (SQLITE_PATH.name + suffix)
            if p.exists():
                p.unlink()
    
    success = migrate_from_chromadb()
    
    if success:
        print("\nMigration completed successfully!")
        print("You can now use the new fast daemon service:")
        print("  brainlayer search 'your query'")
    else:
        print("\nMigration failed or skipped")
        sys.exit(1)


if __name__ == "__main__":
    main()
