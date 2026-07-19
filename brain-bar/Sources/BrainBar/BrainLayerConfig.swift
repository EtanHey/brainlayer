import Foundation

enum BrainLayerEnrichmentMode: String, CaseIterable, Identifiable, Sendable {
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

enum BrainLayerEnrichmentProvider: String, CaseIterable, Identifiable, Sendable {
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

    static var selectableCases: [BrainLayerEnrichmentProvider] {
        allCases.filter(\.isWiredToday)
    }

    var unavailableReason: String? {
        isWiredToday ? nil : "Runtime integration is not available in this build."
    }
}

enum BrainLayerGoogleAPIKey: Equatable, Sendable, CustomStringConvertible, CustomDebugStringConvertible {
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

    var description: String { displayText }

    var debugDescription: String { displayText }

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

enum BrainLayerLaunchdLoadState: Equatable, Sendable {
    case running
    case loaded
    case unloaded
    case unknown
    case probeError(String)

    var title: String {
        switch self {
        case .running: "Running"
        case .loaded: "Loaded"
        case .unloaded: "Unloaded"
        case .unknown: "Unknown"
        case let .probeError(reason): "Unknown — probe error: \(reason)"
        }
    }
}

enum BrainLayerLaunchdJob: String, CaseIterable, Identifiable, Sendable {
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

struct BrainLayerLaunchdJobSetting: Equatable, Sendable {
    var enabled: Bool
    var loadState: BrainLayerLaunchdLoadState
}

struct BrainLayerConfig: Equatable, Sendable {
    var googleAPIKey: BrainLayerGoogleAPIKey
    var systemEnabled: Bool
    var enrichmentEnabled: Bool
    var enrichmentMode: BrainLayerEnrichmentMode
    var enrichmentProvider: BrainLayerEnrichmentProvider
    var enrichmentBackend: String
    var tuningValues: [String: String]
    var launchdJobs: [BrainLayerLaunchdJob: BrainLayerLaunchdJobSetting]

    static let defaultConfig = BrainLayerConfig(
        googleAPIKey: .missing,
        systemEnabled: true,
        enrichmentEnabled: true,
        enrichmentMode: .remote,
        enrichmentProvider: .gemini,
        enrichmentBackend: "gemini",
        tuningValues: BrainLayerEnvDocument.tuningDefaults,
        launchdJobs: Dictionary(
            uniqueKeysWithValues: BrainLayerLaunchdJob.allCases.map {
                ($0, BrainLayerLaunchdJobSetting(enabled: true, loadState: .unknown))
            }
        )
    )

    func persistedValuesEqual(to other: BrainLayerConfig) -> Bool {
        googleAPIKey == other.googleAPIKey &&
            systemEnabled == other.systemEnabled &&
            enrichmentEnabled == other.enrichmentEnabled &&
            enrichmentMode == other.enrichmentMode &&
            enrichmentProvider == other.enrichmentProvider &&
            enrichmentBackend == other.enrichmentBackend &&
            tuningValues == other.tuningValues &&
            Dictionary(uniqueKeysWithValues: launchdJobs.map { ($0.key, $0.value.enabled) }) ==
            Dictionary(uniqueKeysWithValues: other.launchdJobs.map { ($0.key, $0.value.enabled) })
    }
}

enum BrainLayerConfigValidationResult: Equatable, Sendable {
    case passed
    case failed(String)

    var title: String {
        switch self {
        case .passed: "Passed"
        case let .failed(message): "Failed — \(message)"
        }
    }
}

enum BrainLayerConfigValidator {
    static func validate(_ config: BrainLayerConfig) -> BrainLayerConfigValidationResult {
        if config.enrichmentProvider.unavailableReason != nil {
            return .failed(
                "\(config.enrichmentProvider.title) cannot be activated because its runtime integration is unavailable."
            )
        }
        if config.enrichmentBackend.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return .failed("Enrichment backend is required.")
        }
        return .passed
    }
}

struct BrainLayerActiveRuntimeValues: Equatable, Sendable {
    let systemEnabled: Bool
    let enrichmentEnabled: Bool
    let enrichmentMode: BrainLayerEnrichmentMode
    let enrichmentProvider: BrainLayerEnrichmentProvider
    let enrichmentBackend: String

    init(config: BrainLayerConfig) {
        systemEnabled = config.systemEnabled
        enrichmentEnabled = config.enrichmentEnabled
        enrichmentMode = config.enrichmentMode
        enrichmentProvider = config.enrichmentProvider
        enrichmentBackend = config.enrichmentBackend
    }

    func matches(_ config: BrainLayerConfig) -> Bool {
        self == BrainLayerActiveRuntimeValues(config: config)
    }

    var summary: String {
        "\(enrichmentEnabled ? "Enrichment on" : "Enrichment off") · " +
            "\(enrichmentMode.title) · \(enrichmentProvider.title) · \(enrichmentBackend)"
    }
}

enum BrainLayerActiveRuntimeObservation: Equatable, Sendable {
    case observed(BrainLayerActiveRuntimeValues)
    case unknown(String)

    var summary: String {
        switch self {
        case let .observed(values): values.summary
        case let .unknown(reason): "Unknown — \(reason)"
        }
    }
}

protocol BrainLayerActiveRuntimeSampling: Sendable {
    func sample() -> BrainLayerActiveRuntimeObservation
}

struct UnknownBrainLayerActiveRuntimeProvider: BrainLayerActiveRuntimeSampling {
    func sample() -> BrainLayerActiveRuntimeObservation {
        .unknown("Active runtime configuration is not observable.")
    }
}

struct StaticBrainLayerActiveRuntimeProvider: BrainLayerActiveRuntimeSampling {
    let observation: BrainLayerActiveRuntimeObservation

    func sample() -> BrainLayerActiveRuntimeObservation { observation }
}

enum BrainLayerSettingsService: Equatable, Hashable, Sendable {
    case enrichment
    case systemJobs
    case launchdJob(BrainLayerLaunchdJob)

    var title: String {
        switch self {
        case .enrichment: "Enrichment service"
        case .systemJobs: "BrainLayer jobs"
        case let .launchdJob(job): job.launchdLabel
        }
    }
}

enum BrainLayerActiveRuntimeReceiptState: Equatable, Sendable {
    case observed
    case notObserved
    case unknown(String)

    var title: String {
        switch self {
        case .observed: "Observed active"
        case .notObserved: "Not active yet"
        case let .unknown(reason): "Unknown — \(reason)"
        }
    }
}

struct BrainLayerSettingsSaveReceipt: Equatable, Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    let configURL: URL
    let savedAt: Date
    let fileUpdated: Bool
    let validation: BrainLayerConfigValidationResult
    let servicesRequiringRestart: [BrainLayerSettingsService]
    let activeRuntimeState: BrainLayerActiveRuntimeReceiptState

    var description: String {
        let services = servicesRequiringRestart.map(\.title).joined(separator: ", ")
        return "fileUpdated=\(fileUpdated) validation=\(validation.title) " +
            "restart=\(services.isEmpty ? "none" : services) active=\(activeRuntimeState.title)"
    }

    var debugDescription: String { description }
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
        let includeLegacyGoogleKey = originalLines.contains {
            Self.assignmentKey($0) == "GOOGLE_GENERATIVE_AI_API_KEY"
        }
        let managedValues = Self.managedValues(for: config, includeLegacyGoogleKey: includeLegacyGoogleKey)
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

    static let tuningDefaults: [String: String] = [
        "BRAINLAYER_ENRICH_RATE": "15",
        "BRAINLAYER_ENRICH_CONCURRENCY": "4",
        "BRAINLAYER_MAX_COMMIT_BATCH": "25",
        "BRAINLAYER_GEMINI_SERVICE_TIER": "flex",
        "BRAINLAYER_DISABLED_SLEEP_SECONDS": "3600",
    ]

    private static let managedKeyOrder: [String] = [
        "GOOGLE_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
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

    private static let tuningKeyOrder: [String] = [
        "BRAINLAYER_ENRICH_RATE",
        "BRAINLAYER_ENRICH_CONCURRENCY",
        "BRAINLAYER_MAX_COMMIT_BATCH",
        "BRAINLAYER_GEMINI_SERVICE_TIER",
        "BRAINLAYER_DISABLED_SLEEP_SECONDS",
    ]

    private static func managedValues(
        for config: BrainLayerConfig,
        includeLegacyGoogleKey: Bool
    ) -> [String: String] {
        var values: [String: String] = [
            "GOOGLE_API_KEY": config.googleAPIKey.renderedValue,
            "BRAINLAYER_SYSTEM_ENABLED": config.systemEnabled ? "1" : "0",
            "BRAINLAYER_ENRICH_ENABLED": config.enrichmentEnabled ? "1" : "0",
            "BRAINLAYER_ENRICH_MODE": config.enrichmentMode.rawValue,
            "BRAINLAYER_ENRICH_PROVIDER": config.enrichmentProvider.rawValue,
            "BRAINLAYER_ENRICH_BACKEND": config.enrichmentBackend,
        ]
        if includeLegacyGoogleKey {
            values["GOOGLE_GENERATIVE_AI_API_KEY"] = ""
        }
        for key in tuningKeyOrder {
            values[key] = config.tuningValues[key] ?? tuningDefaults[key] ?? ""
        }
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
        for key in tuningKeyOrder {
            guard let raw = assignments[key] else { continue }
            config.tuningValues[key] = raw
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

struct BrainLayerLaunchdCommandResult: Sendable, Equatable {
    let terminationStatus: Int32
    let output: String
}

protocol BrainLayerLaunchdStatusSampling: Sendable {
    func sample() -> [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState]
}

struct BrainLayerLaunchdStatusProvider: BrainLayerLaunchdStatusSampling {
    typealias CommandRunner = @Sendable ([String]) -> BrainLayerLaunchdCommandResult

    private let commandRunner: CommandRunner
    private let uidProvider: @Sendable () -> uid_t

    init(
        commandRunner: @escaping CommandRunner = BrainLayerLaunchdStatusProvider.run,
        uidProvider: @escaping @Sendable () -> uid_t = getuid
    ) {
        self.commandRunner = commandRunner
        self.uidProvider = uidProvider
    }

    func sample() -> [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] {
        Dictionary(
            uniqueKeysWithValues: BrainLayerLaunchdJob.allCases.map { job in
                (job, state(for: job.launchdLabel))
            }
        )
    }

    private func state(for label: String) -> BrainLayerLaunchdLoadState {
        let target = "gui/\(uidProvider())/\(label)"
        let result = commandRunner(["/bin/launchctl", "print", target])
        if result.terminationStatus == 0 {
            return result.output.contains("pid =") ? .running : .loaded
        }

        if result.terminationStatus == 113 ||
            result.output.localizedCaseInsensitiveContains("could not find service") {
            return .unloaded
        }
        return .probeError("launchctl exited \(result.terminationStatus)")
    }

    private static func run(_ command: [String]) -> BrainLayerLaunchdCommandResult {
        guard let executable = command.first else {
            return BrainLayerLaunchdCommandResult(terminationStatus: 1, output: "")
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = Array(command.dropFirst())
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        do {
            try process.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            return BrainLayerLaunchdCommandResult(
                terminationStatus: process.terminationStatus,
                output: String(data: data, encoding: .utf8) ?? ""
            )
        } catch {
            return BrainLayerLaunchdCommandResult(terminationStatus: 1, output: "")
        }
    }
}

struct StaticBrainLayerLaunchdStatusProvider: BrainLayerLaunchdStatusSampling {
    let states: [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState]

    func sample() -> [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] {
        states
    }
}
