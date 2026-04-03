// EntityCache.swift — In-memory entity name cache for sub-millisecond detection.
//
// Loaded at startup from kg_entities, refreshed on a 60-second timer.
// Uses Dictionary<String, String> (lowercased name → entity ID) for O(1) lookup.
// Greedy longest-match for multi-word entity names.

import Foundation
import SQLite3

final class EntityCache: @unchecked Sendable {
    struct DetectedEntity {
        let id: String
        let name: String  // original-case name from DB
    }

    /// lowercased name → (entity ID, original name)
    private var entityMap: [String: (id: String, name: String)] = [:]
    /// Sorted longest-first for greedy multi-word matching
    private var sortedNames: [String] = []
    private let queue = DispatchQueue(label: "com.brainbar.entitycache", attributes: .concurrent)

    private var refreshTimer: DispatchSourceTimer?

    /// Load all entity names from the database.
    func load(from db: OpaquePointer?) {
        guard let db else { return }
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT id, name FROM kg_entities", -1, &stmt, nil) == SQLITE_OK
        else { return }
        defer { sqlite3_finalize(stmt) }

        var newMap: [String: (String, String)] = [:]
        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let idPtr = sqlite3_column_text(stmt, 0),
                  let namePtr = sqlite3_column_text(stmt, 1) else { continue }
            let id = String(cString: idPtr)
            let name = String(cString: namePtr)
            let key = name.lowercased()
            // Keep the first occurrence (avoids overwriting with duplicate lowercase entries)
            if newMap[key] == nil {
                newMap[key] = (id, name)
            }
        }

        var names = Array(newMap.keys)
        names.sort { $0.count > $1.count }  // longest-first for greedy matching

        queue.async(flags: .barrier) {
            self.entityMap = newMap
            self.sortedNames = names
        }
        NSLog("[BrainBar] EntityCache loaded: %d entities", newMap.count)
    }

    /// Start a 60-second periodic refresh timer.
    func startRefreshTimer(db: OpaquePointer?) {
        // Cancel any existing timer before creating a new one
        refreshTimer?.cancel()
        refreshTimer = nil

        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + 60, repeating: 60)
        timer.setEventHandler { [weak self] in
            self?.load(from: db)
        }
        timer.resume()
        refreshTimer = timer
    }

    func stopRefreshTimer() {
        refreshTimer?.cancel()
        refreshTimer = nil
    }

    /// Detect known entity names in a search query.
    /// Returns detected entities ordered by match priority (multi-word first, then single-word).
    func detectEntities(in query: String) -> [DetectedEntity] {
        let queryLower = query.lowercased()
        let words = queryLower.split(separator: " ").map(String.init)
        var results: [DetectedEntity] = []
        var matched = Set<String>()

        queue.sync {
            // Multi-word entities first (greedy, longest match)
            for key in sortedNames where key.contains(" ") && queryLower.contains(key) {
                if let entry = entityMap[key], !matched.contains(key) {
                    results.append(DetectedEntity(id: entry.id, name: entry.name))
                    matched.insert(key)
                }
            }
            // Single-word entities
            for word in words {
                if let entry = entityMap[word], !matched.contains(word) {
                    results.append(DetectedEntity(id: entry.id, name: entry.name))
                    matched.insert(word)
                }
            }
        }
        return results
    }

    var count: Int {
        queue.sync { entityMap.count }
    }
}
