import Foundation
import SQLite3

private func require(_ condition: @autoclosure () -> Bool, _ message: String) {
    if !condition() { fatalError(message) }
}

private func text(_ value: Any?) -> String? { value as? String }

@main
struct NativeHeaderPublicJoinHarness {
    static func main() {
        guard CommandLine.arguments.count >= 3 else { fatalError("usage: harness db record-key [expected-status]") }
        let path = CommandLine.arguments[1]
        let key = CommandLine.arguments[2]
        var db: OpaquePointer?
        require(sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX, nil) == SQLITE_OK, "open")
        defer { sqlite3_close(db) }
        let result = NativeContinuityReader.readTargetRecord(db: db, sourceRecordKey: key, maxBytes: 32 * 1024)
        print("result=\(result)")
        let expected = CommandLine.arguments.count > 3 ? CommandLine.arguments[3] : "available"
        require(result["status"] as? String == expected, "public reader status \(result)")
        if expected != "available" {
            if let attribution = result["attribution"] as? [String: Any],
               let emitter = attribution["recorded_emitter"] as? [String: Any] {
                require(emitter["principal"] is NSNull, "tainted emitter principal")
                require(emitter["recorded_thread_id"] == nil || emitter["recorded_thread_id"] is NSNull, "tainted emitter thread leaked")
                require(emitter["recorded_agent_path"] == nil || emitter["recorded_agent_path"] is NSNull, "tainted emitter path leaked")
            }
            print("native header public join: PASS status=\(expected)"); return
        }
        let header = result["native_header"] as? [String: Any] ?? [:]
        require(header["status"] as? String == "available", "header status")
        require(header["provider_recorded"] as? Bool == true, "provider evidence")
        require(header["declared_unverified"] as? Bool == true, "unverified evidence")
        require(header["recorded_thread_id"] as? String == "01a08d1e-42f6-7093-83a4-1c1b160f4911", "thread id")
        require(header["recorded_agent_path"] as? String == "/worker" && header["recorded_agent_nickname"] as? String == "Luna", "payload metadata")
        require(header["spawn_agent_path"] as? String == "/spawned" && header["spawn_agent_nickname"] as? String == "SpawnLuna", "nested spawn metadata")
        require(header["person_principal"] is NSNull && header["original_author_principal"] is NSNull, "principal boundary")
        let attribution = result["attribution"] as? [String: Any] ?? [:]
        let emitter = attribution["recorded_emitter"] as? [String: Any] ?? [:]
        require(emitter["principal"] is NSNull, "attribution principal boundary")
        require(emitter["principal_status"] as? String == "unknown_no_distinct_identity_binding", "principal status")
        require(emitter["status"] as? String == "declared_fields_no_principal_binding", "emitter declaration status")
        require(emitter["provider_recorded"] as? Bool == true, "emitter provider evidence")
        require(emitter["declared_unverified"] as? Bool == true, "emitter unverified evidence")
        require(emitter["evidence_kind"] as? String == "provider_recorded_header", "emitter evidence kind")
        require((emitter["evidence_keys"] as? [String]) == [key], "emitter evidence key")
        require(emitter["recorded_thread_id"] as? String == "01a08d1e-42f6-7093-83a4-1c1b160f4911", "emitter thread id")
        require(emitter["recorded_agent_path"] as? String == "/worker", "emitter agent path")
        require(emitter["recorded_agent_nickname"] as? String == "Luna", "emitter agent nickname")
        require(emitter["spawn_agent_path"] as? String == "/spawned", "emitter spawn path")
        require(emitter["spawn_agent_nickname"] as? String == "SpawnLuna", "emitter spawn nickname")
        require((attribution["original_content_author"] as? [String: Any])?["principal"] is NSNull, "attribution author boundary")
        print("native header public join: PASS status=available source_record_key=\(key)")
    }
}
