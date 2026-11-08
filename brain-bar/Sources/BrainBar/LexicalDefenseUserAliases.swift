import Foundation

/// Optional per-user lexical alias source for the native search fallback (#995).
///
/// Same file, bounds and conflict policy as `src/brainlayer/lexical_defense.py`, so the hybrid
/// helper path and this fallback expand the same user aliases. The file lives outside the app and
/// package; personal terms are never shipped. Missing file -> no user aliases (behavior unchanged).
/// A malformed, oversized or unreadable file is ignored whole and logged once per file version.
/// Freshness: the file is stat()ed on every lookup and re-read only when (mtime, size) changes.
final class LexicalDefenseUserAliases: @unchecked Sendable {
    static let environmentKey = "BRAINLAYER_LEXICAL_DEFENSE_LOCAL"
    static let schemaVersion = 1
    static let maxBytes = 64 * 1024
    static let maxEntries = 256
    static let maxSurfacesPerEntry = 16
    static let maxSurfaceChars = 64

    struct ConfigError: Error, CustomStringConvertible {
        let description: String
    }

    struct Parsed: Equatable {
        var replacements: [String: [String]] = [:]
        var skipped: [String] = []
    }

    private struct Stamp: Equatable {
        let modified: Date?
        let size: Int
    }

    let url: URL
    private let claimed: Set<String>
    private let lock = NSLock()
    private var cache: (stamp: Stamp?, parsed: Parsed)?

    init(url: URL = LexicalDefenseUserAliases.defaultURL(), claimedSurfaces: Set<String> = []) {
        self.url = url
        self.claimed = Set(claimedSurfaces.map(Self.normalize))
    }

    static func defaultURL(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> URL {
        if let raw = environment[environmentKey]?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty {
            return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath)
        }
        return homeDirectory.appendingPathComponent(".config/brainlayer/lexical_defense.local.json")
    }

    /// Letters and digits only, compatibility-normalized and case-folded: the lookup key on both paths.
    static func normalize(_ value: String) -> String {
        let scalars = value.precomposedStringWithCompatibilityMapping
            .folding(options: [.caseInsensitive], locale: Locale(identifier: "en_US_POSIX")).unicodeScalars
        return String(String.UnicodeScalarView(scalars.filter { CharacterSet.alphanumerics.contains($0) }))
    }

    func replacements(for token: String) -> [String] {
        let key = Self.normalize(token)
        guard !key.isEmpty else { return [] }
        return current().replacements[key] ?? []
    }

    private func current() -> Parsed {
        let stamp = currentStamp()
        lock.lock()
        defer { lock.unlock() }
        if let cached = cache, cached.stamp == stamp { return cached.parsed }
        var parsed = Parsed()
        if stamp != nil {
            do {
                let handle = try FileHandle(forReadingFrom: url)
                defer { try? handle.close() }
                parsed = try Self.parse(handle.read(upToCount: Self.maxBytes + 1) ?? Data(), claimed: claimed)
                if !parsed.skipped.isEmpty {
                    NSLog("[BrainBar] lexical defense: %d user alias entries skipped on conflict: %@", parsed.skipped.count, url.path)
                }
            } catch {
                NSLog("[BrainBar] lexical defense: ignoring user alias file %@: %@", url.path, String(describing: error))
            }
        }
        cache = (stamp, parsed)
        return parsed
    }

    private func currentStamp() -> Stamp? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path) else {
            return Stamp(modified: nil, size: -1)
        }
        return Stamp(
            modified: attributes[.modificationDate] as? Date,
            size: (attributes[.size] as? NSNumber)?.intValue ?? -1
        )
    }

    /// Strict schema; any violation rejects the WHOLE file. A user entry any of whose surfaces is
    /// already claimed (packaged, or an earlier user entry) is skipped whole, which also breaks
    /// every cross-entry cycle. Expansion is one hop, never transitive.
    static func parse(_ data: Data, claimed: Set<String> = []) throws -> Parsed {
        guard data.count <= maxBytes else { throw ConfigError(description: "file exceeds \(maxBytes) bytes") }
        guard String(data: data, encoding: .utf8) != nil,
              let object = try? JSONSerialization.jsonObject(with: data),
              let root = object as? [String: Any]
        else { throw ConfigError(description: "not a UTF-8 JSON object") }
        guard Set(root.keys) == ["version", "entries"] else {
            throw ConfigError(description: "top level must be exactly {version, entries}")
        }
        guard let version = root["version"] as? NSNumber,
              CFGetTypeID(version) != CFBooleanGetTypeID(),
              version.doubleValue == Double(schemaVersion)
        else { throw ConfigError(description: "version must be \(schemaVersion)") }
        guard let items = root["entries"] as? [Any], items.count <= maxEntries else {
            throw ConfigError(description: "entries must be a list of at most \(maxEntries)")
        }

        var entries: [[String]] = []
        for (index, item) in items.enumerated() {
            let place = "entries[\(index)]"
            guard let entry = item as? [String: Any], let canonical = entry["canonical"],
                  Set(entry.keys).isSubset(of: ["canonical", "aliases", "split_forms"])
            else { throw ConfigError(description: "\(place): keys must be canonical plus optional aliases/split_forms") }
            var surfaces = try [surface(canonical, "\(place).canonical")]
            var extra = 0
            for key in ["aliases", "split_forms"] {
                guard let values = (entry[key] ?? [Any]()) as? [Any] else {
                    throw ConfigError(description: "\(place).\(key): must be a list")
                }
                extra += values.count
                surfaces += try values.map { try surface($0, "\(place).\(key)") }
            }
            guard extra <= maxSurfacesPerEntry else {
                throw ConfigError(description: "\(place): more than \(maxSurfacesPerEntry) surfaces")
            }
            entries.append(surfaces)
        }

        var parsed = Parsed()
        var taken = claimed
        for surfaces in entries {
            var seen = Set<String>()
            let unique = surfaces.filter { seen.insert($0).inserted }
            let keys = Set(unique.map(normalize))
            guard keys.isDisjoint(with: taken) else {
                parsed.skipped.append(unique[0])
                continue
            }
            taken.formUnion(keys)
            for key in keys { parsed.replacements[key] = unique }
        }
        return parsed
    }

    private static func surface(_ value: Any, _ place: String) throws -> String {
        guard let raw = value as? String else { throw ConfigError(description: "\(place): surface must be a string") }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard (1...maxSurfaceChars).contains(trimmed.unicodeScalars.count), !normalize(trimmed).isEmpty else {
            throw ConfigError(description: "\(place): surface must be 1...\(maxSurfaceChars) chars with a letter or digit")
        }
        return trimmed
    }
}
