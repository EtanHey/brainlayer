import Foundation
import GRDB

@MainActor
final class InjectionStore: ObservableObject {
    @Published private(set) var events: [InjectionEvent] = []

    private let dbQueue: DatabaseQueue
    private var cancellable: AnyDatabaseCancellable?

    init(databasePath: String) throws {
        dbQueue = try DatabaseQueue(path: databasePath)
    }

    func start(sessionID: String? = nil, limit: Int = 50) {
        let observation = ValueObservation.tracking { db -> [InjectionEvent] in
            let sql: String
            let arguments: StatementArguments
            if let sessionID {
                sql = """
                    SELECT id, session_id, timestamp, query, chunk_ids, token_count
                    FROM injection_events
                    WHERE session_id = ?
                    ORDER BY timestamp DESC, id DESC
                    LIMIT ?
                """
                arguments = [sessionID, limit]
            } else {
                sql = """
                    SELECT id, session_id, timestamp, query, chunk_ids, token_count
                    FROM injection_events
                    ORDER BY timestamp DESC, id DESC
                    LIMIT ?
                """
                arguments = [limit]
            }

            let rows = try Row.fetchAll(db, sql: sql, arguments: arguments)
            return try rows.map { row in
                try InjectionEvent(
                    row: [
                        "id": row["id"] as Int64,
                        "session_id": row["session_id"] as String,
                        "timestamp": row["timestamp"] as String,
                        "query": row["query"] as String,
                        "chunk_ids": row["chunk_ids"] as String,
                        "token_count": Int(row["token_count"] as Int64)
                    ]
                )
            }
        }

        cancellable = observation.start(
            in: dbQueue,
            onError: { error in
                NSLog("[BrainBar] InjectionStore observation error: %@", String(describing: error))
            },
            onChange: { [weak self] events in
                self?.events = events
            }
        )
    }

    func stop() {
        cancellable?.cancel()
        cancellable = nil
    }
}
