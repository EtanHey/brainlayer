ALTER TABLE chunks ADD COLUMN half_life_days REAL DEFAULT 30.0;
ALTER TABLE chunks ADD COLUMN last_retrieved REAL DEFAULT NULL;
ALTER TABLE chunks ADD COLUMN retrieval_count INTEGER DEFAULT 0;
ALTER TABLE chunks ADD COLUMN decay_score REAL DEFAULT 1.0;
ALTER TABLE chunks ADD COLUMN pinned INTEGER DEFAULT 0;
ALTER TABLE chunks ADD COLUMN archived INTEGER DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_chunks_decay_score ON chunks(decay_score) WHERE archived = 0;
CREATE INDEX IF NOT EXISTS idx_chunks_last_retrieved ON chunks(last_retrieved) WHERE archived = 0;
