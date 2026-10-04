import Foundation

/// A BrainLayer launchd job's effective environment, built the way `brainlayer-env-run.sh` builds
/// it: the installed plist's `EnvironmentVariables`, then the env file it names (else BrainBar's
/// `BRAINLAYER_ENV_FILE`, else the default) exported over them. BrainBar's own environment only
/// fills keys the job does not set.
enum BrainLayerJobEnvironment {
    static func effective(
        plist: Data?,
        brainBarEnvironment: [String: String],
        home: URL,
        readFile: (URL) -> Data?
    ) -> [String: String] {
        let agent = plist
            .flatMap { try? PropertyListSerialization.propertyList(from: $0, format: nil) as? [String: Any] }
        var job = (agent?["EnvironmentVariables"] as? [String: Any])?.compactMapValues { $0 as? String } ?? [:]
        let envFile = [job["BRAINLAYER_ENV_FILE"], brainBarEnvironment["BRAINLAYER_ENV_FILE"]]
            .lazy.compactMap { $0 }.first { !$0.isEmpty }
            .map { expandTilde($0, home: home) }
            ?? home.appendingPathComponent(".config/brainlayer/brainlayer.env").path
        if let text = readFile(URL(fileURLWithPath: envFile)).flatMap({ String(data: $0, encoding: .utf8) }) {
            for (key, value) in envFileValues(text) { job[key] = value }
        }
        return brainBarEnvironment.merging(job) { _, jobValue in jobValue }
    }

    static func expandTilde(_ path: String, home: URL) -> String {
        path == "~" ? home.path : path.hasPrefix("~/") ? home.path + path.dropFirst() : path
    }

    /// The simple `KEY=value` / `export KEY="value"` lines `brainlayer-env-run.sh` exports; command
    /// substitutions are skipped there too.
    static func envFileValues(_ text: String) -> [String: String] {
        var values: [String: String] = [:]
        for raw in text.split(whereSeparator: \.isNewline) {
            var line = raw.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty, !line.hasPrefix("#") else { continue }
            if line.hasPrefix("export ") { line = String(line.dropFirst("export ".count)) }
            guard let equals = line.firstIndex(of: "=") else { continue }
            let key = line[..<equals].trimmingCharacters(in: .whitespaces)
            var value = line[line.index(after: equals)...].trimmingCharacters(in: .whitespaces)
            guard !key.isEmpty, !value.contains("$("), !value.contains("`") else { continue }
            if value.count >= 2, let first = value.first, first == value.last, first == "\"" || first == "'" {
                value = String(value.dropFirst().dropLast())
            }
            values[key] = value
        }
        return values
    }
}
