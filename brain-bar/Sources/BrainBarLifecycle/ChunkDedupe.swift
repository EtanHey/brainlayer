import CryptoKit
import Foundation

public enum ChunkDedupe {
    private static let stopwords: Set<String> = [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on",
        "or", "that", "the", "this", "to", "was", "were", "with",
    ]
    private static let tokenPattern = try! NSRegularExpression(pattern: "[a-z0-9]+")
    private static let isoTimestampPattern = try! NSRegularExpression(
        pattern: #"\b\d{4}-\d{2}-\d{2}[T ][0-2]\d:[0-5]\d:[0-5]\d(?:\.\d+)?(?:Z|[+-][0-2]\d:?[0-5]\d)?\b"#,
        options: [.caseInsensitive]
    )
    private static let shingleWidth = 4
    private static let simhashBits = 64

    public struct Fields {
        public let dedupeHash: String
        public let simhash: String
        public let bands: (String, String, String, String)
    }

    public static func computeDedupeFields(content: String, createdAt: String? = nil) -> Fields {
        let simhashHex = simhashHex64(content: content, createdAt: createdAt)
        return Fields(
            dedupeHash: normalizedExactHash(content),
            simhash: simhashHex,
            bands: (
                String(simhashHex.prefix(4)),
                String(simhashHex.dropFirst(4).prefix(4)),
                String(simhashHex.dropFirst(8).prefix(4)),
                String(simhashHex.dropFirst(12).prefix(4))
            )
        )
    }

    static func normalizedExactHash(_ content: String) -> String {
        let normalized = normalizeForExactHash(content)
        let digest = SHA256.hash(data: Data(normalized.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private static func normalizeForExactHash(_ content: String) -> String {
        tokens(from: content.lowercased()).joined(separator: " ")
    }

    private static func normalizeForDedupe(_ content: String) -> String {
        let lowered = content.lowercased()
        let range = NSRange(lowered.startIndex..<lowered.endIndex, in: lowered)
        let stripped = isoTimestampPattern.stringByReplacingMatches(
            in: lowered,
            range: range,
            withTemplate: " "
        )
        return tokens(from: stripped).joined(separator: " ")
    }

    private static func tokens(from text: String) -> [String] {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return tokenPattern.matches(in: text, range: range).compactMap { match in
            guard let tokenRange = Range(match.range, in: text) else { return nil }
            let token = String(text[tokenRange])
            return stopwords.contains(token) ? nil : token
        }
    }

    private static func simhashHex64(content: String, createdAt: String?) -> String {
        let fingerprint = simhash64(content: content, createdAt: createdAt)
        return String(format: "%016llx", fingerprint)
    }

    private static func simhash64(content: String, createdAt: String?) -> UInt64 {
        let normalized = normalizeForDedupe(content)
        let tokens = normalized.split(separator: " ").map(String.init)
        var weights = [Double](repeating: 0, count: simhashBits)
        let features: [(String, Double)]
        if tokens.isEmpty {
            return 0
        } else if tokens.count < shingleWidth {
            features = tokens.map { ("tok:\($0)", 1.0) }
        } else {
            features = (0...(tokens.count - shingleWidth)).map { index in
                let shingle = tokens[index..<(index + shingleWidth)].joined(separator: " ")
                return ("sh:\(shingle)", 1.0)
            }
        }
        for (feature, weight) in features {
            let hashed = stableU64(feature)
            for bit in 0..<simhashBits {
                if hashed & (1 << bit) != 0 {
                    weights[bit] += weight
                } else {
                    weights[bit] -= weight
                }
            }
        }
        var fingerprint: UInt64 = 0
        for (bit, weight) in weights.enumerated() {
            if weight >= 0 {
                fingerprint |= 1 << bit
            }
        }
        return fingerprint
    }

    private static func stableU64(_ feature: String) -> UInt64 {
        let digest = SHA256.hash(data: Data(feature.utf8))
        return digest.prefix(8).reduce(UInt64(0)) { partial, byte in
            (partial << 8) | UInt64(byte)
        }
    }
}
