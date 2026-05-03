// MCPFraming.swift — MCP Content-Length framing parser.
//
// MCP uses Content-Length headers (like LSP):
//   Content-Length: N\r\n\r\n{json payload of exactly N bytes}
//
// This struct accumulates bytes and extracts complete JSON-RPC messages.

import Foundation

struct MCPFraming: Sendable {
    private var buffer = Data()
    private static let separator = Data("\r\n\r\n".utf8)
    /// Current buffer size — for debug logging.
    var bufferCount: Int { buffer.count }
    /// Set to false when messages are extracted via the raw JSON fallback.
    /// Tells the server to respond without Content-Length framing.
    private(set) var lastExtractUsedContentLength: Bool = true
    /// Max payload size (10 MB) — prevents DoS via absurd Content-Length values.
    private static let maxContentLength = 10_000_000

    /// Max total buffer size (16 MB) — prevents memory exhaustion from slow/malicious clients.
    private static let maxBufferSize = 16_000_000

    /// Append raw bytes from a socket read. Drops data if buffer would exceed limit.
    mutating func append(_ data: Data) {
        guard buffer.count + data.count <= Self.maxBufferSize else {
            buffer.removeAll()
            return
        }
        buffer.append(data)
    }

    /// Extract all complete messages from the buffer.
    /// Incomplete messages remain in the buffer for the next append.
    ///
    /// Supports two modes:
    /// 1. Standard MCP Content-Length framing (preferred).
    /// 2. Raw JSON-RPC fallback — if no Content-Length header is found,
    ///    tries to parse the buffer as bare JSON. This prevents silent
    ///    hangs when a client omits framing (e.g. socat manual tests,
    ///    or a transport that strips headers).
    mutating func extractMessages() -> [[String: Any]] {
        var messages: [[String: Any]] = []

        while true {
            // Find the header/body separator
            guard let separatorRange = buffer.range(of: Self.separator) else {
                break
            }

            // Parse Content-Length from headers
            let headerData = buffer[buffer.startIndex..<separatorRange.lowerBound]
            guard let headerStr = String(data: headerData, encoding: .utf8),
                  let contentLength = parseContentLength(headerStr),
                  contentLength > 0, contentLength <= Self.maxContentLength else {
                // Invalid or zero-length — skip past this separator
                buffer = Data(buffer[separatorRange.upperBound...])
                continue
            }

            let bodyStart = separatorRange.upperBound

            // Check if we have enough body bytes
            guard buffer.count >= bodyStart + contentLength else {
                // Incomplete body — wait for more data
                break
            }

            // Extract body
            let bodyData = buffer[bodyStart..<(bodyStart + contentLength)]
            buffer = Data(buffer[(bodyStart + contentLength)...])

            // Parse JSON
            if let json = try? JSONSerialization.jsonObject(with: bodyData) as? [String: Any] {
                messages.append(json)
                lastExtractUsedContentLength = true
            }
        }

        // Fallback: raw JSON-RPC without Content-Length framing.
        // Only triggers when the buffer has data, no \r\n\r\n separator was
        // found, and no Content-Length messages were already extracted.
        if messages.isEmpty && !buffer.isEmpty && buffer.range(of: Self.separator) == nil {
            // Try newline-delimited JSON first (handles echo piped input)
            let newline = UInt8(0x0A)
            var cursor = buffer.startIndex
            while cursor < buffer.endIndex {
                guard let nlIndex = buffer[cursor...].firstIndex(of: newline) else { break }
                let lineData = buffer[cursor..<nlIndex]
                cursor = buffer.index(after: nlIndex)
                guard !lineData.isEmpty,
                      let json = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any],
                      json["method"] is String else { continue }
                messages.append(json)
            }
            // Try remaining data after last newline (no trailing newline)
            if cursor < buffer.endIndex {
                let tail = buffer[cursor...]
                if let json = try? JSONSerialization.jsonObject(with: tail) as? [String: Any],
                   json["method"] is String {
                    messages.append(json)
                    cursor = buffer.endIndex
                }
            }
            if !messages.isEmpty {
                buffer = cursor < buffer.endIndex ? Data(buffer[cursor...]) : Data()
                lastExtractUsedContentLength = false
            }
        }

        return messages
    }

    /// Encode a JSON-RPC response with Content-Length framing.
    static func encode(_ response: [String: Any]) throws -> Data {
        let jsonData = try encodeJSONResponse(response)
        let header = "Content-Length: \(jsonData.count)\r\n\r\n"
        var frame = Data(header.utf8)
        frame.append(jsonData)
        return frame
    }

    /// Encode JSON-RPC envelopes with a deterministic top-level key order.
    ///
    /// Claude Desktop currently rejects otherwise-valid JSON-RPC objects when
    /// Foundation serializes `result` or `id` before `jsonrpc`.
    static func encodeJSONResponse(_ response: [String: Any]) throws -> Data {
        guard let jsonrpc = response["jsonrpc"],
              let id = response["id"],
              response["result"] != nil || response["error"] != nil else {
            return try JSONSerialization.data(withJSONObject: response)
        }

        var data = Data("{\"jsonrpc\":".utf8)
        data.append(try encodeJSONValue(jsonrpc))
        data.append(Data(",\"id\":".utf8))
        data.append(try encodeJSONValue(id))
        if let result = response["result"] {
            data.append(Data(",\"result\":".utf8))
            data.append(try encodeJSONValue(result))
        } else if let error = response["error"] {
            data.append(Data(",\"error\":".utf8))
            data.append(try encodeJSONValue(error))
        }
        data.append(0x7D) // }
        return data
    }

    // MARK: - Private

    private static func encodeJSONValue(_ value: Any) throws -> Data {
        try JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed])
    }

    private func parseContentLength(_ header: String) -> Int? {
        for line in header.split(separator: "\r\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.lowercased().hasPrefix("content-length:") {
                let value = trimmed.dropFirst("content-length:".count).trimmingCharacters(in: .whitespaces)
                return Int(value)
            }
        }
        return nil
    }
}
