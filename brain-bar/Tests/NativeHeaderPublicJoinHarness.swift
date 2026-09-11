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
        if expected != "available" { print("native header public join: PASS status=\(expected)"); return }
        let header = result["native_header"] as? [String: Any] ?? [:]
        require(header["status"] as? String == "available", "header status")
        require(header["provider_recorded"] as? Bool == true, "provider evidence")
        require(header["declared_unverified"] as? Bool == true, "unverified evidence")
        require(header["recorded_thread_id"] as? String == "01a08d1e-42f6-7093-83a4-1c1b160f4911", "thread id")
        require(header["person_principal"] is NSNull && header["original_author_principal"] is NSNull, "principal boundary")
        let attribution = result["attribution"] as? [String: Any] ?? [:]
        let emitter = attribution["recorded_emitter"] as? [String: Any] ?? [:]
        require(emitter["principal"] is NSNull, "attribution principal boundary")
        print("native header public join: PASS status=available source_record_key=\(key)")
    }
}
