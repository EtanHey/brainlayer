// BrainDatabase.swift — SQLite database layer for BrainBar.
//
// Wraps SQLite3 C API directly (no external dependencies).
// Configures: WAL mode, FTS5, busy_timeout=5000, cache_size=-64000, synchronous=NORMAL.
// Single-writer: only BrainBar writes. Concurrent reads are safe (WAL).

import Foundation
import SQLite3

final class BrainDatabase: @unchecked Sendable {
    private var db: OpaquePointer?
    private let path: String
    private let queue = DispatchQueue(label: "com.brainlayer.brainbar.db")

    init(path: String) {
        self.path = path
        openAndConfigure()
    }

    private func openAndConfigure() {
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        let rc = sqlite3_open_v2(path, &db, flags, nil)
        guard rc == SQLITE_OK else {
            NSLog("[BrainBar] Failed to open database: %d", rc)
            return
        }

        // Configure PRAGMAs
        exec("PRAGMA journal_mode = WAL")
        exec("PRAGMA busy_timeout = 5000")
        exec("PRAGMA cache_size = -64000")
        exec("PRAGMA synchronous = NORMAL")

        // Create schema
        createSchema()
    }

    private func createSchema() {
        // Main chunks table
        exec("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                session_id TEXT,
                project TEXT,
                content TEXT NOT NULL,
                content_type TEXT DEFAULT 'assistant_text',
                importance INTEGER DEFAULT 5,
                tags TEXT DEFAULT '[]',
                source TEXT DEFAULT 'claude_code',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                summary TEXT,
                intent TEXT,
                sentiment TEXT
            )
        """)

        // FTS5 virtual table for full-text search
        exec("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                summary,
                tags,
                content='chunks',
                content_rowid='rowid'
            )
        """)

        // FTS sync triggers
        exec("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content, summary, tags)
                VALUES (new.rowid, new.content, new.summary, new.tags);
            END
        """)

        exec("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, summary, tags)
                VALUES ('delete', old.rowid, old.content, old.summary, old.tags);
            END
        """)

        exec("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, summary, tags)
                VALUES ('delete', old.rowid, old.content, old.summary, old.tags);
                INSERT INTO chunks_fts(rowid, content, summary, tags)
                VALUES (new.rowid, new.content, new.summary, new.tags);
            END
        """)
    }

    func close() {
        if let db {
            sqlite3_close(db)
            self.db = nil
        }
    }

    // MARK: - PRAGMA queries

    func pragma(_ name: String) throws -> String {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, "PRAGMA \(name)", -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        guard sqlite3_step(stmt) == SQLITE_ROW else { throw DBError.noResult }
        guard let cStr = sqlite3_column_text(stmt, 0) else { throw DBError.noResult }
        return String(cString: cStr)
    }

    // MARK: - Table existence

    func tableExists(_ name: String) throws -> Bool {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let sql = "SELECT count(*) FROM sqlite_master WHERE type IN ('table','view') AND name = ?"
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, name, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
        guard sqlite3_step(stmt) == SQLITE_ROW else { return false }
        return sqlite3_column_int(stmt, 0) > 0
    }

    // MARK: - Insert chunk

    func insertChunk(id: String, content: String, sessionId: String, project: String, contentType: String, importance: Int) throws {
        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let sql = """
            INSERT OR REPLACE INTO chunks (chunk_id, content, session_id, project, content_type, importance)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(stmt, 1, id, -1, TRANSIENT)
        sqlite3_bind_text(stmt, 2, content, -1, TRANSIENT)
        sqlite3_bind_text(stmt, 3, sessionId, -1, TRANSIENT)
        sqlite3_bind_text(stmt, 4, project, -1, TRANSIENT)
        sqlite3_bind_text(stmt, 5, contentType, -1, TRANSIENT)
        sqlite3_bind_int(stmt, 6, Int32(importance))

        let stepRC = sqlite3_step(stmt)
        guard stepRC == SQLITE_DONE else { throw DBError.step(stepRC) }
    }

    // MARK: - FTS5 Search

    func search(query: String, limit: Int) throws -> [[String: Any]] {
        guard let db else { throw DBError.notOpen }

        // Sanitize query for FTS5 — escape double quotes, wrap tokens in quotes
        let sanitized = sanitizeFTS5Query(query)

        var stmt: OpaquePointer?
        let sql = """
            SELECT c.chunk_id, c.content, c.project, c.content_type, c.importance,
                   c.created_at, c.summary, c.tags, c.session_id
            FROM chunks_fts f
            JOIN chunks c ON f.rowid = c.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(stmt, 1, sanitized, -1, TRANSIENT)
        sqlite3_bind_int(stmt, 2, Int32(limit))

        var results: [[String: Any]] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            var row: [String: Any] = [:]
            row["chunk_id"] = columnText(stmt, 0)
            row["content"] = columnText(stmt, 1)
            row["project"] = columnText(stmt, 2)
            row["content_type"] = columnText(stmt, 3)
            row["importance"] = Int(sqlite3_column_int(stmt, 4))
            row["created_at"] = columnText(stmt, 5)
            row["summary"] = columnText(stmt, 6)
            row["tags"] = columnText(stmt, 7)
            row["session_id"] = columnText(stmt, 8)
            results.append(row)
        }

        return results
    }

    // MARK: - Store (brain_store)

    func store(content: String, tags: [String], importance: Int, source: String) throws -> String {
        let id = "brainbar-\(UUID().uuidString.lowercased().prefix(12))"
        let tagsJSON: String
        if let data = try? JSONSerialization.data(withJSONObject: tags),
           let str = String(data: data, encoding: .utf8) {
            tagsJSON = str
        } else {
            tagsJSON = "[]"
        }

        guard let db else { throw DBError.notOpen }
        var stmt: OpaquePointer?
        let sql = """
            INSERT INTO chunks (chunk_id, content, tags, importance, source, content_type)
            VALUES (?, ?, ?, ?, ?, 'user_message')
        """
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else { throw DBError.prepare(rc) }
        defer { sqlite3_finalize(stmt) }

        let TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(stmt, 1, id, -1, TRANSIENT)
        sqlite3_bind_text(stmt, 2, content, -1, TRANSIENT)
        sqlite3_bind_text(stmt, 3, tagsJSON, -1, TRANSIENT)
        sqlite3_bind_int(stmt, 4, Int32(importance))
        sqlite3_bind_text(stmt, 5, source, -1, TRANSIENT)

        let stepRC = sqlite3_step(stmt)
        guard stepRC == SQLITE_DONE else { throw DBError.step(stepRC) }

        return id
    }

    // MARK: - Helpers

    private func exec(_ sql: String) {
        guard let db else { return }
        var errMsg: UnsafeMutablePointer<CChar>?
        let rc = sqlite3_exec(db, sql, nil, nil, &errMsg)
        if rc != SQLITE_OK {
            let msg = errMsg.map { String(cString: $0) } ?? "unknown error"
            NSLog("[BrainBar] SQL error: %@ (code: %d)", msg, rc)
            sqlite3_free(errMsg)
        }
    }

    private func columnText(_ stmt: OpaquePointer?, _ col: Int32) -> String? {
        guard let cStr = sqlite3_column_text(stmt, col) else { return nil }
        return String(cString: cStr)
    }

    private func sanitizeFTS5Query(_ query: String) -> String {
        // Split into tokens, quote each one for safety
        let tokens = query.split(separator: " ").map { token -> String in
            let cleaned = token.replacingOccurrences(of: "\"", with: "")
            return "\"\(cleaned)\""
        }
        return tokens.joined(separator: " OR ")
    }

    // MARK: - Errors

    enum DBError: LocalizedError {
        case notOpen
        case prepare(Int32)
        case step(Int32)
        case noResult

        var errorDescription: String? {
            switch self {
            case .notOpen: return "Database not open"
            case .prepare(let rc): return "SQLite prepare failed: \(rc)"
            case .step(let rc): return "SQLite step failed: \(rc)"
            case .noResult: return "No result"
            }
        }
    }

    deinit {
        close()
    }
}
