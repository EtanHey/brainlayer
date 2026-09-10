import Foundation
@testable import BrainBar

private func require(_ condition: @autoclosure () -> Bool, _ message: String) {
    if !condition() { fatalError(message) }
}

private func listedTools(_ router: MCPRouter) -> [[String: Any]] {
    let response = router.handle([
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
    ])
    return (response["result"] as? [String: Any])?["tools"] as? [[String: Any]] ?? []
}

private func containsSchemaKey(_ value: Any, _ key: String) -> Bool {
    if let object = value as? [String: Any] {
        return object[key] != nil || object.values.contains { containsSchemaKey($0, key) }
    }
    if let array = value as? [Any] {
        return array.contains { containsSchemaKey($0, key) }
    }
    return false
}

private func json(_ value: Any) -> Data {
    try! JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
}

private func errorText(_ response: [String: Any]) -> String {
    let result = response["result"] as? [String: Any] ?? [:]
    let content = result["content"] as? [[String: Any]] ?? []
    return content.compactMap { $0["text"] as? String }.joined(separator: "\n")
}

private func call(_ router: MCPRouter, name: String, arguments: [String: Any]) -> [String: Any] {
    router.handle([
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": ["name": name, "arguments": arguments],
    ])
}

@main
struct CorePaletteBudgetHarness {
    static func main() throws {
        let core = listedTools(MCPRouter(profile: "core"))
        let full = listedTools(MCPRouter(profile: "full"))
        let coreBytes = json(core).count
        print("core tools/list bytes: \(coreBytes)")
        require(coreBytes <= 1_600, "core tools/list exceeds 1600 bytes: \(coreBytes)")

        let coreByName = Dictionary(uniqueKeysWithValues: core.compactMap { tool in
            (tool["name"] as? String).map { ($0, tool) }
        })
        let fullByName = Dictionary(uniqueKeysWithValues: full.compactMap { tool in
            (tool["name"] as? String).map { ($0, tool) }
        })
        for name in ["brain_search", "brain_store", "brain_recall", "brain_expand"] {
            let compact = coreByName[name]?["inputSchema"] as? [String: Any] ?? [:]
            let canonical = fullByName[name]?["inputSchema"] as? [String: Any] ?? [:]
            require(!containsSchemaKey(compact, "maxLength"), "\(name) compact maxLength leaked")
            require(!containsSchemaKey(compact, "maxItems"), "\(name) compact maxItems leaked")
            require(containsSchemaKey(canonical, "maxLength"), "\(name) canonical maxLength missing")
            let compactProperties = Set((compact["properties"] as? [String: Any] ?? [:]).keys)
            let canonicalProperties = Set((canonical["properties"] as? [String: Any] ?? [:]).keys)
            require(compactProperties == canonicalProperties, "\(name) property discoverability changed")
        }
        let compactSearch = coreByName["brain_search"]?["inputSchema"] as? [String: Any] ?? [:]
        let fullSearch = fullByName["brain_search"]?["inputSchema"] as? [String: Any] ?? [:]
        let compactSearchProperties = compactSearch["properties"] as? [String: Any] ?? [:]
        let fullSearchProperties = fullSearch["properties"] as? [String: Any] ?? [:]
        let compactSourceEnum = (compactSearchProperties["source"] as? [String: Any])?["enum"] as? [String]
        let fullSourceEnum = (fullSearchProperties["source"] as? [String: Any])?["enum"] as? [String]
        require(compactSourceEnum == fullSourceEnum, "source enum changed")
        let compactExpand = coreByName["brain_expand"]?["inputSchema"] as? [String: Any] ?? [:]
        let fullExpand = fullByName["brain_expand"]?["inputSchema"] as? [String: Any] ?? [:]
        require(
            json(compactExpand["oneOf"] as? [[String: Any]] ?? [])
                == json(fullExpand["oneOf"] as? [[String: Any]] ?? []),
            "expand XOR changed"
        )
        let compactExpandProperties = compactExpand["properties"] as? [String: Any] ?? [:]
        let fullExpandProperties = fullExpand["properties"] as? [String: Any] ?? [:]
        require(
            (compactExpandProperties["source_record_key"] as? [String: Any])?["pattern"] as? String
                == (fullExpandProperties["source_record_key"] as? [String: Any])?["pattern"] as? String,
            "source key regex changed"
        )
        require(coreByName["brain_store"]?["description"] as? String ==
            "Store memory. status: STORED|DUPLICATE|MERGED|DEFERRED all = success, do NOT re-store; REJECTED|ERROR = nothing stored, no chunk_id.",
            "brain_store outcome contract changed")

        let router = MCPRouter(profile: "core")
        let limits: [(String, [String: Any], String)] = [
            ("brain_search", ["query": String(repeating: "q", count: 4_097)], "maxLength 4096"),
            ("brain_store", ["content": String(repeating: "c", count: 200_001)], "maxLength 200000"),
            ("brain_expand", ["chunk_id": String(repeating: "i", count: 4_097)], "maxLength 4096"),
            ("brain_store", ["content": "x", "tags": Array(repeating: "tag", count: 101)], "maxItems 100"),
        ]
        for (name, arguments, expected) in limits {
            let response = call(router, name: name, arguments: arguments)
            require((response["result"] as? [String: Any])?["isError"] as? Bool == true, "\(name) limit accepted")
            require(errorText(response).contains(expected), "\(name) missing \(expected)")
        }
        print("core palette budget harness: PASS")
    }
}
