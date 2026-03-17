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
            }
        }

        return messages
    }

    /// Encode a JSON-RPC response with Content-Length framing.
    static func encode(_ response: [String: Any]) throws -> Data {
        let jsonData = try JSONSerialization.data(withJSONObject: response)
        let header = "Content-Length: \(jsonData.count)\r\n\r\n"
        var frame = Data(header.utf8)
        frame.append(jsonData)
        return frame
    }

    // MARK: - Private

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
