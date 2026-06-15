import Foundation

enum BrainLayerEnrichmentMode: String, CaseIterable, Identifiable {
    case remote
    case local

    var id: String { rawValue }

    var title: String {
        switch self {
        case .remote: "Remote"
        case .local: "Local"
        }
    }
}

enum BrainLayerEnrichmentProvider: String, CaseIterable, Identifiable {
    case gemini
    case openai
    case anthropic

    var id: String { rawValue }

    var title: String {
        switch self {
        case .gemini: "Gemini"
        case .openai: "OpenAI"
        case .anthropic: "Anthropic"
        }
    }

    var isWiredToday: Bool {
        self == .gemini
    }
}

enum BrainLayerGoogleAPIKey: Equatable {
    enum Kind: Equatable {
        case missing
        case plainPresent
        case onePasswordReference
    }

    case missing
    case plain(String)
    case onePasswordReference(String)

    var kind: Kind {
        switch self {
        case .missing: .missing
        case .plain: .plainPresent
        case .onePasswordReference: .onePasswordReference
        }
    }

    var displayText: String {
        switch self {
        case .missing: "Not configured"
        case .plain: "Stored in config file"
        case .onePasswordReference: "1Password reference"
        }
    }

    var opReference: String {
        if case let .onePasswordReference(reference) = self {
            return reference
        }
        return "op://Private/Google AI/Gemini API key"
    }

    fileprivate var renderedValue: String {
        switch self {
        case .missing:
            ""
        case let .plain(value):
            Self.shellSingleQuoted(value)
        case let .onePasswordReference(reference):
            "\"$(op read \(Self.shellSingleQuoted(reference)))\""
        }
    }

    private static func shellSingleQuoted(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }
}

enum BrainLayerLaunchdLoadState: Equatable {
    case running
    case loaded
    case unloaded
    case unknown

    var title: String {
        switch self {
        case .running: "Running"
        case .loaded: "Loaded"
        case .unloaded: "Unloaded"
        case .unknown: "Unknown"
        }
    }
}

enum BrainLayerLaunchdJob: String, CaseIterable, Identifiable {
    case enrichment
    case hotlane
    case decay
    case drain
    case watch
    case index
    case backupDaily
    case jsonlBackup
    case maintenanceNightly
    case maintenanceWeekly
    case repairFTS
    case walCheckpoint

    var id: String { rawValue }

    var title: String {
        switch self {
        case .enrichment: "Enrichment"
        case .hotlane: "Hotlane"
        case .decay: "Decay"
        case .drain: "Drain"
        case .watch: "Watch"
        case .index: "Index"
        case .backupDaily: "Backup Daily"
        case .jsonlBackup: "JSONL Backup"
        case .maintenanceNightly: "Maintenance Nightly"
        case .maintenanceWeekly: "Maintenance Weekly"
        case .repairFTS: "Repair FTS"
        case .walCheckpoint: "WAL Checkpoint"
        }
    }

    var configKey: String {
        switch self {
        case .enrichment: "BRAINLAYER_LAUNCHD_ENRICHMENT_ENABLED"
        case .hotlane: "BRAINLAYER_LAUNCHD_HOTLANE_ENABLED"
        case .decay: "BRAINLAYER_LAUNCHD_DECAY_ENABLED"
        case .drain: "BRAINLAYER_LAUNCHD_DRAIN_ENABLED"
        case .watch: "BRAINLAYER_LAUNCHD_WATCH_ENABLED"
        case .index: "BRAINLAYER_LAUNCHD_INDEX_ENABLED"
        case .backupDaily: "BRAINLAYER_LAUNCHD_BACKUP_DAILY_ENABLED"
        case .jsonlBackup: "BRAINLAYER_LAUNCHD_JSONL_BACKUP_ENABLED"
        case .maintenanceNightly: "BRAINLAYER_LAUNCHD_MAINTENANCE_NIGHTLY_ENABLED"
        case .maintenanceWeekly: "BRAINLAYER_LAUNCHD_MAINTENANCE_WEEKLY_ENABLED"
        case .repairFTS: "BRAINLAYER_LAUNCHD_REPAIR_FTS_ENABLED"
        case .walCheckpoint: "BRAINLAYER_LAUNCHD_WAL_CHECKPOINT_ENABLED"
        }
    }

    var launchdLabel: String {
        switch self {
        case .backupDaily: "com.brainlayer.backup-daily"
        case .jsonlBackup: "com.brainlayer.jsonl-backup"
        case .maintenanceNightly: "com.brainlayer.maintenance-nightly"
        case .maintenanceWeekly: "com.brainlayer.maintenance-weekly"
        case .repairFTS: "com.brainlayer.repair-fts"
        case .walCheckpoint: "com.brainlayer.wal-checkpoint"
        default: "com.brainlayer.\(rawValue)"
        }
    }
}

struct BrainLayerLaunchdJobSetting: Equatable {
    var enabled: Bool
    var loadState: BrainLayerLaunchdLoadState
}

struct BrainLayerConfig: Equatable {
    var googleAPIKey: BrainLayerGoogleAPIKey
    var systemEnabled: Bool
    var enrichmentEnabled: Bool
    var enrichmentMode: BrainLayerEnrichmentMode
    var enrichmentProvider: BrainLayerEnrichmentProvider
    var enrichmentBackend: String
    var launchdJobs: [BrainLayerLaunchdJob: BrainLayerLaunchdJobSetting]

    static let defaultConfig = BrainLayerConfig(
        googleAPIKey: .missing,
        systemEnabled: true,
        enrichmentEnabled: true,
        enrichmentMode: .remote,
        enrichmentProvider: .gemini,
        enrichmentBackend: "gemini",
        launchdJobs: Dictionary(
            uniqueKeysWithValues: BrainLayerLaunchdJob.allCases.map {
                ($0, BrainLayerLaunchdJobSetting(enabled: true, loadState: .unknown))
            }
        )
    )
}

struct BrainLayerEnvDocument {
    private var originalLines: [String]
    private(set) var config: BrainLayerConfig

    init(text: String) throws {
        originalLines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        config = Self.parse(lines: originalLines)
    }

    init(config: BrainLayerConfig) {
        originalLines = []
        self.config = config
    }

    mutating func update(_ apply: (inout BrainLayerConfig) -> Void) {
        apply(&config)
    }

    func rendered() -> String {
        let managedValues = Self.managedValues(for: config)
        var emitted = Set<String>()
        var output: [String] = []

        for line in originalLines {
            guard let key = Self.assignmentKey(line), managedValues[key] != nil else {
                output.append(line)
                continue
            }
            guard !emitted.contains(key), let value = managedValues[key] else {
                continue
            }
            output.append("\(key)=\(value)")
            emitted.insert(key)
        }

        if output.isEmpty {
            output.append("# BrainLayer private config.")
        }
        if !output.isEmpty, output.last != "" {
            output.append("")
        }

        for key in Self.managedKeyOrder where !emitted.contains(key) {
            guard let value = managedValues[key] else { continue }
            output.append("\(key)=\(value)")
            emitted.insert(key)
        }

        return output.joined(separator: "\n") + "\n"
    }

    private static let managedKeyOrder: [String] = [
        "GOOGLE_API_KEY",
        "BRAINLAYER_SYSTEM_ENABLED",
        "BRAINLAYER_ENRICH_ENABLED",
        "BRAINLAYER_ENRICH_MODE",
        "BRAINLAYER_ENRICH_PROVIDER",
        "BRAINLAYER_ENRICH_BACKEND",
        "BRAINLAYER_ENRICH_RATE",
        "BRAINLAYER_ENRICH_CONCURRENCY",
        "BRAINLAYER_MAX_COMMIT_BATCH",
        "BRAINLAYER_GEMINI_SERVICE_TIER",
        "BRAINLAYER_DISABLED_SLEEP_SECONDS",
        "BRAINLAYER_LAUNCHD_ENRICHMENT_ENABLED",
        "BRAINLAYER_LAUNCHD_HOTLANE_ENABLED",
        "BRAINLAYER_LAUNCHD_DECAY_ENABLED",
        "BRAINLAYER_LAUNCHD_DRAIN_ENABLED",
        "BRAINLAYER_LAUNCHD_WATCH_ENABLED",
        "BRAINLAYER_LAUNCHD_INDEX_ENABLED",
        "BRAINLAYER_LAUNCHD_BACKUP_DAILY_ENABLED",
        "BRAINLAYER_LAUNCHD_JSONL_BACKUP_ENABLED",
        "BRAINLAYER_LAUNCHD_MAINTENANCE_NIGHTLY_ENABLED",
        "BRAINLAYER_LAUNCHD_MAINTENANCE_WEEKLY_ENABLED",
        "BRAINLAYER_LAUNCHD_REPAIR_FTS_ENABLED",
        "BRAINLAYER_LAUNCHD_WAL_CHECKPOINT_ENABLED",
    ]

    private static func managedValues(for config: BrainLayerConfig) -> [String: String] {
        var values: [String: String] = [
            "GOOGLE_API_KEY": config.googleAPIKey.renderedValue,
            "BRAINLAYER_SYSTEM_ENABLED": config.systemEnabled ? "1" : "0",
            "BRAINLAYER_ENRICH_ENABLED": config.enrichmentEnabled ? "1" : "0",
            "BRAINLAYER_ENRICH_MODE": config.enrichmentMode.rawValue,
            "BRAINLAYER_ENRICH_PROVIDER": config.enrichmentProvider.rawValue,
            "BRAINLAYER_ENRICH_BACKEND": config.enrichmentBackend,
            "BRAINLAYER_ENRICH_RATE": "15",
            "BRAINLAYER_ENRICH_CONCURRENCY": "4",
            "BRAINLAYER_MAX_COMMIT_BATCH": "25",
            "BRAINLAYER_GEMINI_SERVICE_TIER": "flex",
            "BRAINLAYER_DISABLED_SLEEP_SECONDS": "3600",
        ]
        for job in BrainLayerLaunchdJob.allCases {
            values[job.configKey] = config.launchdJobs[job]?.enabled == false ? "0" : "1"
        }
        return values
    }

    private static func parse(lines: [String]) -> BrainLayerConfig {
        var config = BrainLayerConfig.defaultConfig
        let assignments = Dictionary(lines.compactMap(Self.assignment), uniquingKeysWith: { _, new in new })

        if let rawKey = assignments["GOOGLE_API_KEY"] ?? assignments["GOOGLE_GENERATIVE_AI_API_KEY"] {
            config.googleAPIKey = parseGoogleKey(rawKey)
        }
        if let raw = assignments["BRAINLAYER_SYSTEM_ENABLED"] {
            config.systemEnabled = !isFalse(raw)
        }
        if let raw = assignments["BRAINLAYER_ENRICH_ENABLED"] {
            config.enrichmentEnabled = !isFalse(raw)
        }
        if let raw = assignments["BRAINLAYER_ENRICH_MODE"],
           let mode = BrainLayerEnrichmentMode(rawValue: normalized(raw)) {
            config.enrichmentMode = mode
        }
        if let raw = assignments["BRAINLAYER_ENRICH_PROVIDER"],
           let provider = BrainLayerEnrichmentProvider(rawValue: normalized(raw)) {
            config.enrichmentProvider = provider
        }
        if let raw = assignments["BRAINLAYER_ENRICH_BACKEND"], !normalized(raw).isEmpty {
            config.enrichmentBackend = normalized(raw)
        }
        for job in BrainLayerLaunchdJob.allCases {
            guard let raw = assignments[job.configKey] else { continue }
            config.launchdJobs[job]?.enabled = !isFalse(raw)
        }
        return config
    }

    private static func parseGoogleKey(_ raw: String) -> BrainLayerGoogleAPIKey {
        let value = strippedValue(raw)
        guard !value.isEmpty else { return .missing }
        if value.contains("op read") {
            return .onePasswordReference(extractOpReference(from: value) ?? "op://Private/Google AI/Gemini API key")
        }
        return .plain(value)
    }

    private static func extractOpReference(from value: String) -> String? {
        guard let range = value.range(of: "op://") else { return nil }
        let suffix = value[range.lowerBound...]
        let terminators = CharacterSet(charactersIn: "'\")")
        if let end = suffix.unicodeScalars.firstIndex(where: { terminators.contains($0) }) {
            return String(String.UnicodeScalarView(suffix.unicodeScalars[..<end]))
        }
        return String(suffix)
    }

    private static func assignment(_ line: String) -> (String, String)? {
        guard let key = assignmentKey(line) else { return nil }
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        let withoutExport = trimmed.hasPrefix("export ") ? String(trimmed.dropFirst("export ".count)) : trimmed
        let parts = withoutExport.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2 else { return nil }
        return (key, String(parts[1]).trimmingCharacters(in: .whitespaces))
    }

    private static func assignmentKey(_ line: String) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else { return nil }
        let withoutExport = trimmed.hasPrefix("export ") ? String(trimmed.dropFirst("export ".count)) : trimmed
        guard let equals = withoutExport.firstIndex(of: "=") else { return nil }
        return String(withoutExport[..<equals]).trimmingCharacters(in: .whitespaces)
    }

    private static func isFalse(_ value: String) -> Bool {
        ["0", "false", "no", "off", "disabled"].contains(normalized(value))
    }

    private static func normalized(_ value: String) -> String {
        strippedValue(value).lowercased()
    }

    private static func strippedValue(_ value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespaces)
        if trimmed.count >= 2,
           let first = trimmed.first,
           let last = trimmed.last,
           (first == "'" && last == "'") || (first == "\"" && last == "\"") {
            return String(trimmed.dropFirst().dropLast())
        }
        return trimmed
    }
}

struct BrainLayerConfigStore {
    let configURL: URL
    var fileManager: FileManager = .default

    static func defaultConfigURL(homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
        homeDirectory
            .appendingPathComponent(".config", isDirectory: true)
            .appendingPathComponent("brainlayer", isDirectory: true)
            .appendingPathComponent("brainlayer.env", isDirectory: false)
    }

    init(configURL: URL = BrainLayerConfigStore.defaultConfigURL()) {
        self.configURL = configURL
    }

    func loadDocument() throws -> BrainLayerEnvDocument {
        guard fileManager.fileExists(atPath: configURL.path) else {
            return BrainLayerEnvDocument(config: .defaultConfig)
        }
        let text = try String(contentsOf: configURL, encoding: .utf8)
        return try BrainLayerEnvDocument(text: text)
    }

    func save(_ config: BrainLayerConfig) throws {
        var document = try loadDocument()
        document.update { $0 = config }
        try fileManager.createDirectory(at: configURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try document.rendered().write(to: configURL, atomically: true, encoding: .utf8)
        try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: configURL.path)
    }
}

protocol BrainLayerLaunchdStatusSampling {
    func sample() -> [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState]
}

struct BrainLayerLaunchdStatusProvider: BrainLayerLaunchdStatusSampling {
    func sample() -> [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] {
        Dictionary(
            uniqueKeysWithValues: BrainLayerLaunchdJob.allCases.map { job in
                (job, state(for: job.launchdLabel))
            }
        )
    }

    private func state(for label: String) -> BrainLayerLaunchdLoadState {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["print", "gui/\(getuid())/\(label)"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        do {
            try process.run()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else { return .unloaded }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: data, encoding: .utf8) ?? ""
            return output.contains("pid =") ? .running : .loaded
        } catch {
            return .unknown
        }
    }
}

struct StaticBrainLayerLaunchdStatusProvider: BrainLayerLaunchdStatusSampling {
    let states: [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState]

    func sample() -> [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] {
        states
    }
}
