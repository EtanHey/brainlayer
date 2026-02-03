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
CHROMADB_PATH = Path.home() / ".local" / "share" / "zikaron" / "chromadb"
SQLITE_PATH = Path.home() / ".local" / "share" / "zikaron" / "zikaron.db"


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
        
        # Get existing collection
        try:
            collection = client.get_collection("conversations")
        except Exception:
            print("No 'conversations' collection found in ChromaDB")
            return False
        
        # Get all data
        print("Fetching all data from ChromaDB...")
        all_data = collection.get(include=["documents", "metadatas", "embeddings"])
        
        if not all_data["ids"]:
            print("No data found in ChromaDB")
            return False
        
        total_chunks = len(all_data["ids"])
        print(f"Found {total_chunks} chunks to migrate")
        
        # Prepare data for sqlite-vec
        chunks = []
        embeddings = []
        
        for i, chunk_id in enumerate(all_data["ids"]):
            document = all_data["documents"][i]
            metadata = all_data["metadatas"][i] or {}
            embedding = all_data["embeddings"][i] if all_data["embeddings"] else None
            
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
            
            # Handle embedding dimension mismatch
            if embedding:
                if len(embedding) == 768:  # nomic-embed-text dimension
                    # Need to re-embed with bge-small-en-v1.5 (384 dim)
                    embedding = None
                elif len(embedding) == 384:  # Already correct dimension
                    embeddings.append(embedding)
                else:
                    # Unknown dimension, re-embed
                    embedding = None
            
            if embedding is None:
                embeddings.append(None)  # Will re-embed later
        
        # Create sqlite-vec store
        print(f"Creating sqlite-vec database at {SQLITE_PATH}")
        vector_store = VectorStore(SQLITE_PATH)
        
        # Re-embed chunks that need it
        need_embedding = [i for i, emb in enumerate(embeddings) if emb is None]
        
        if need_embedding:
            print(f"Re-embedding {len(need_embedding)} chunks with bge-small-en-v1.5...")
            embedding_model = get_embedding_model()
            
            # Re-embed in batches
            batch_size = 32
            for i in range(0, len(need_embedding), batch_size):
                batch_indices = need_embedding[i:i + batch_size]
                batch_texts = [chunks[idx]["content"] for idx in batch_indices]
                
                try:
                    batch_embeddings = embedding_model._load_model().encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                    
                    for j, idx in enumerate(batch_indices):
                        embeddings[idx] = batch_embeddings[j].tolist()
                    
                    print(f"Re-embedded {min(i + batch_size, len(need_embedding))}/{len(need_embedding)} chunks")
                    
                except Exception as e:
                    logger.error(f"Failed to re-embed batch: {e}")
                    # Use zero vector as fallback
                    for idx in batch_indices:
                        embeddings[idx] = [0.0] * 384
        
        # Insert data in batches
        batch_size = 1000
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            vector_store.upsert_chunks(batch_chunks, batch_embeddings)
            print(f"Migrated {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
        
        vector_store.close()
        
        # Verify migration
        vector_store = VectorStore(SQLITE_PATH)
        final_count = vector_store.count()
        vector_store.close()
        
        print(f"Migration complete: {final_count} chunks in sqlite-vec")
        
        # Backup ChromaDB
        backup_path = CHROMADB_PATH.parent / "chromadb.backup"
        if not backup_path.exists():
            print(f"Backing up ChromaDB to {backup_path}")
            import shutil
            shutil.move(str(CHROMADB_PATH), str(backup_path))
            print("ChromaDB backed up successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        print(f"Migration failed: {e}")
        return False


def main():
    """Main migration entry point."""
    logging.basicConfig(level=logging.INFO)
    
    print("Zikaron Migration Tool")
    print("=" * 50)
    
    if SQLITE_PATH.exists():
        response = input(f"sqlite-vec database already exists at {SQLITE_PATH}. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Migration cancelled")
            return
        
        # Remove existing database
        SQLITE_PATH.unlink()
    
    success = migrate_from_chromadb()
    
    if success:
        print("\nMigration completed successfully!")
        print("You can now use the new fast daemon service:")
        print("  zikaron search 'your query'")
    else:
        print("\nMigration failed or skipped")
        sys.exit(1)


if __name__ == "__main__":
    main()
