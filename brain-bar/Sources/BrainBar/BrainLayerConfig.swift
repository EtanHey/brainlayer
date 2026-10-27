import Foundation

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
    case ownerUnknown(String)
    case paused
    case failed(String)

    var title: String {
        switch self {
        case .running: "Running"
        case .loaded: "Loaded"
        case .unloaded: "Unloaded"
        case .unknown: "Unknown"
        case let .probeError(reason): "Unknown — probe error: \(reason)"
        case let .ownerUnknown(reason): reason
        case .paused: "Paused"
        case let .failed(reason): "Failed — \(reason)"
        }
    }
}

enum BrainLayerLaunchdJob: String, CaseIterable, Identifiable, Sendable {
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
        case .hotlane: "com.brainlayer.hotlane-brainbar"
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
    var launchdJobs: [BrainLayerLaunchdJob: BrainLayerLaunchdJobSetting]

    var enrichmentIsOff: Bool {
        true // Enrichment is retired; legacy config flags cannot enable it.
    }

    static let defaultConfig = BrainLayerConfig(
        googleAPIKey: .missing,
        systemEnabled: true,
        launchdJobs: Dictionary(
            uniqueKeysWithValues: BrainLayerLaunchdJob.allCases.map {
                ($0, BrainLayerLaunchdJobSetting(enabled: true, loadState: .unknown))
            }
        )
    )

    func persistedValuesEqual(to other: BrainLayerConfig) -> Bool {
        googleAPIKey == other.googleAPIKey &&
            systemEnabled == other.systemEnabled &&
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

struct BrainLayerActiveRuntimeValues: Equatable, Sendable {
    let systemEnabled: Bool

    init(config: BrainLayerConfig) {
        systemEnabled = config.systemEnabled
    }

    func matches(_ config: BrainLayerConfig) -> Bool {
        self == BrainLayerActiveRuntimeValues(config: config)
    }

    var summary: String {
        "BrainLayer jobs \(systemEnabled ? "on" : "off")"
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
    case systemJobs
    case launchdJob(BrainLayerLaunchdJob)

    var title: String {
        switch self {
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

    private static let managedKeyOrder: [String] = [
        "GOOGLE_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "BRAINLAYER_SYSTEM_ENABLED",
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

    private static func managedValues(
        for config: BrainLayerConfig,
        includeLegacyGoogleKey: Bool
    ) -> [String: String] {
        var values: [String: String] = [
            "GOOGLE_API_KEY": config.googleAPIKey.renderedValue,
            "BRAINLAYER_SYSTEM_ENABLED": config.systemEnabled ? "1" : "0",
        ]
        if includeLegacyGoogleKey {
            values["GOOGLE_GENERATIVE_AI_API_KEY"] = ""
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
        for job in BrainLayerLaunchdJob.allCases {
            guard let raw = assignments[job.configKey] else { continue }
            config.launchdJobs[job]?.enabled = !isFalse(raw)
        }
        return config
    }

    private static func parseGoogleKey(_ raw: String) -> BrainLayerGoogleAPIKey {
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        let value: String
        if trimmed.count >= 2, trimmed.first == "'", trimmed.last == "'" {
            value = String(trimmed.dropFirst().dropLast())
                .replacingOccurrences(of: "'\"'\"'", with: "'")
        } else {
            value = strippedValue(raw)
        }
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

    private static func isTrue(_ value: String) -> Bool {
        ["1", "true"].contains(normalized(value))
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
    private let loadDocumentOverride: (() throws -> BrainLayerEnvDocument)?
    private let saveOverride: ((BrainLayerConfig) throws -> Void)?

    static func defaultConfigURL(homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
        homeDirectory
            .appendingPathComponent(".config", isDirectory: true)
            .appendingPathComponent("brainlayer", isDirectory: true)
            .appendingPathComponent("brainlayer.env", isDirectory: false)
    }

    init(
        configURL: URL = BrainLayerConfigStore.defaultConfigURL(),
        fileManager: FileManager = .default,
        loadDocumentOverride: (() throws -> BrainLayerEnvDocument)? = nil,
        saveOverride: ((BrainLayerConfig) throws -> Void)? = nil
    ) {
        self.configURL = configURL
        self.fileManager = fileManager
        self.loadDocumentOverride = loadDocumentOverride
        self.saveOverride = saveOverride
    }

    func loadDocument() throws -> BrainLayerEnvDocument {
        if let loadDocumentOverride {
            return try loadDocumentOverride()
        }
        guard fileManager.fileExists(atPath: configURL.path) else {
            return BrainLayerEnvDocument(config: .defaultConfig)
        }
        let text = try String(contentsOf: configURL, encoding: .utf8)
        return try BrainLayerEnvDocument(text: text)
    }

    func save(_ config: BrainLayerConfig) throws {
        if let saveOverride {
            try saveOverride(config)
            return
        }
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
    func sampleActivity() -> [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation]
}

extension BrainLayerLaunchdStatusSampling {
    func sampleActivity() -> [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation] {
        sample().mapValues(BrainLayerLaunchdJobObservation.stateOnly)
    }
}

struct BrainLayerLaunchdStatusProvider: BrainLayerLaunchdStatusSampling {
    typealias CommandRunner = @Sendable ([String]) -> BrainLayerLaunchdCommandResult

    private let commandRunner: CommandRunner
    private let uidProvider: @Sendable () -> uid_t
    private let runRecordDate: @Sendable (BrainLayerLaunchdJob) -> Date?
    private let schedule: @Sendable (BrainLayerLaunchdJob) -> DateComponents?
    private let calendar: Calendar
    private let now: @Sendable () -> Date
    private let ownership: @Sendable () -> BrainBarJobOwnership
    private let calendarHistory: @Sendable () -> BrainBarCalendarHistory
    private let calendarSchedule: @Sendable () -> DateComponents?
    private let calendarScheduleProblem: @Sendable () -> String?
    private let calendarLogsDirectory: @Sendable () -> URL
    private let calendarIsDisabled: @Sendable (BrainLayerLaunchdJob) -> Bool

    init(
        commandRunner: @escaping CommandRunner = BrainLayerLaunchdStatusProvider.run,
        uidProvider: @escaping @Sendable () -> uid_t = getuid,
        runRecordDate: @escaping @Sendable (BrainLayerLaunchdJob) -> Date? = { readRunRecord($0) },
        schedule: @escaping @Sendable (BrainLayerLaunchdJob) -> DateComponents? = { readSchedule($0) },
        calendar: Calendar = .current,
        now: @escaping @Sendable () -> Date = Date.init,
        ownership: @escaping @Sendable () -> BrainBarJobOwnership = { .read() },
        calendarHistory: @escaping @Sendable () -> BrainBarCalendarHistory = { .read() },
        calendarSchedule: @escaping @Sendable () -> DateComponents? = { readSchedule(label: "com.brainlayer.maintenance") },
        calendarScheduleProblem: @escaping @Sendable () -> String? = { scheduleProblem(label: "com.brainlayer.maintenance") },
        calendarLogsDirectory: @escaping @Sendable () -> URL = { readCalendarLogsDirectory() },
        calendarIsDisabled: @escaping @Sendable (BrainLayerLaunchdJob) -> Bool = {
            (try? BrainLayerConfigStore().loadDocument())?.config.launchdJobs[$0]?.enabled == false
        }
    ) {
        self.ownership = ownership
        self.calendarHistory = calendarHistory
        self.calendarSchedule = calendarSchedule
        self.calendarScheduleProblem = calendarScheduleProblem
        self.calendarLogsDirectory = calendarLogsDirectory
        self.calendarIsDisabled = calendarIsDisabled
        self.commandRunner = commandRunner
        self.uidProvider = uidProvider
        self.runRecordDate = runRecordDate
        self.schedule = schedule
        self.calendar = calendar
        self.now = now
    }

    func sample() -> [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] {
        sampleActivity().mapValues(\.loadState)
    }

    func sampleActivity() -> [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation] {
        switch ownership() {
        case .legacy:
            var results = Dictionary(uniqueKeysWithValues: BrainLayerLaunchdJob.allCases.map { ($0, observation(for: $0)) })
            if let watcher = results[.watch] {
                let daemon = commandRunner(["/bin/launchctl", "print", "gui/\(uidProvider())/com.brainlayer.daemon"])
                results[.watch] = addingDaemonExit(to: watcher, daemon: daemon)
            }
            return results
        case let .unknown(reason):
            return Dictionary(uniqueKeysWithValues: BrainLayerLaunchdJob.allCases.map {
                var result = BrainLayerLaunchdJobObservation.stateOnly(.ownerUnknown(reason))
                result.ownership = .unknown(reason)
                return ($0, result)
            })
        case .supervisor:
            let parent = commandRunner(["/bin/launchctl", "print", "gui/\(uidProvider())/com.brainlayer.maintenance"])
            let daemon = commandRunner(["/bin/launchctl", "print", "gui/\(uidProvider())/com.brainlayer.daemon"])
            let history = calendarHistory()
            let disabledResult = commandRunner(["/bin/launchctl", "print-disabled", "gui/\(uidProvider())"])
            let disabledLabels: Set<String> = {
                guard disabledResult.terminationStatus == 0 else { return [] }
                let pattern = #"["']([^"']+)["']\s*=>\s*(?:true|disabled)\b"#
                guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
                let text = disabledResult.output as NSString
                return Set(regex.matches(in: disabledResult.output, range: NSRange(location: 0, length: text.length))
                    .map { text.substring(with: $0.range(at: 1)) })
            }()
            let next = nextRun(components: calendarSchedule())
            let problem = calendarScheduleProblem()
            return Dictionary(uniqueKeysWithValues: BrainLayerLaunchdJob.allCases.map { job in
                var result = job.isContinuous ? component(job)
                    : calendarStep(job, parent: parent, history: history, next: next, problem: problem)
                if !job.isContinuous {
                    if disabledResult.terminationStatus != 0 {
                        // S5: an unreadable disabled list is uncertainty, never a reason to forget a run.
                        if result.lastRunAt == nil, result.calendarCoveredByWeeklyAt == nil {
                            result = .stateOnly(.probeError("Calendar disabled state unavailable"))
                            result.calendarScheduleProblem = problem
                        } else {
                            result.calendarScheduleUncertainty = result.calendarScheduleUncertainty ?? "Calendar disabled state unavailable"
                        }
                    } else {
                        result.calendarDisabled = result.calendarDisabled || disabledLabels.contains(job.launchdLabel)
                            || disabledLabels.contains("com.brainlayer.maintenance")
                    }
                }
                if job == .maintenanceNightly || job == .maintenanceWeekly {
                    result.calendarNightOutcome = history.maintenanceNight
                }
                if job == .watch {
                    if case .probeError = result.loadState { result.killCheckUnavailable = "Watcher supervisor status unavailable" }
                    if let code = result.lastExitCode, ![0, -15, -2].contains(code) {
                        result.uncleanExit = "Watcher exited uncleanly (exit \(code))"
                        result.healthySince = result.loadState == .running ? result.lastRunAt : nil
                    }
                    result = addingDaemonExit(to: result, daemon: daemon)
                }
                result.ownership = .supervisor
                return (job, result)
            })
        }
    }

    private func addingDaemonExit(to observation: BrainLayerLaunchdJobObservation,
                                  daemon: BrainLayerLaunchdCommandResult) -> BrainLayerLaunchdJobObservation {
        var result = observation
        if let problem = killCheckProblem(daemon) {
            result.killCheckUnavailable = [observation.killCheckUnavailable, problem].compactMap { $0 }.joined(separator: "; ")
            return result
        }
        guard let exit = uncleanLaunchdExit(daemon) else { return result }
        let daemonStart = launchdHealthySince(daemon)
        if let watcherExit = observation.uncleanExit {
            result.uncleanExit = "Supervisor daemon " + exit + "; " + watcherExit
            result.healthySince = if let daemonStart, let watcherStart = observation.healthySince {
                max(daemonStart, watcherStart)
            } else { nil }
        } else {
            result.uncleanExit = "Supervisor daemon " + exit
            result.healthySince = daemonStart
        }
        return result
    }

    private func killCheckProblem(_ result: BrainLayerLaunchdCommandResult) -> String? {
        guard result.terminationStatus == 0 else { return "launchctl print exited \(result.terminationStatus)" }
        if integerLine(named: "pid", in: result.output) != nil || integerLine(named: "runs", in: result.output) != nil
            || integerLine(named: "last exit code", in: result.output) != nil
            || result.output.range(of: #"(?m)^\s*state\s*=\s*\S+"#, options: .regularExpression) != nil
            || firstInteger(matching: #"last terminating signal\s*=\s*[^\n:]*:\s*(\d+)"#, in: result.output) != nil { return nil }
        return "launchctl print output could not be parsed"
    }

    private func uncleanLaunchdExit(_ result: BrainLayerLaunchdCommandResult) -> String? {
        guard result.terminationStatus == 0 else { return nil }
        if let signal = firstInteger(matching: #"last terminating signal\s*=\s*[^\n:]*:\s*(\d+)"#, in: result.output),
           ![0, 2, 15].contains(signal) { return "was killed (signal \(signal))" }
        if let code = integerLine(named: "last exit code", in: result.output), code != 0 {
            return "exited uncleanly (exit \(code))"
        }
        return nil
    }

    private func launchdHealthySince(_ result: BrainLayerLaunchdCommandResult) -> Date? {
        guard let pid = integerLine(named: "pid", in: result.output), pid > 1 else { return nil }
        // BSD ps etime is [[dd-]hh:]mm:ss. Elapsed time survives an app restart and avoids
        // treating a newly observed old kill as a new incident.
        let elapsed = commandRunner(["/bin/ps", "-p", String(pid), "-o", "etime="])
        guard elapsed.terminationStatus == 0 else { return nil }
        let fields = elapsed.output.trimmingCharacters(in: .whitespacesAndNewlines).split(separator: "-")
        guard fields.count <= 2, let clock = fields.last else { return nil }
        let parts = clock.split(separator: ":").compactMap { Double($0) }
        guard !parts.isEmpty, parts.count <= 3, parts.allSatisfy({ $0 >= 0 }),
              parts.count == clock.split(separator: ":").count else { return nil }
        let days = fields.count == 2 ? Double(fields[0]) : 0
        guard let days, days >= 0 else { return nil }
        let seconds = parts.reduce(0) { $0 * 60 + $1 } + days * 86400
        return now().addingTimeInterval(-seconds)
    }

    private func observation(for job: BrainLayerLaunchdJob) -> BrainLayerLaunchdJobObservation {
        let target = "gui/\(uidProvider())/\(job.launchdLabel)"
        let result = commandRunner(["/bin/launchctl", "print", target])
        if result.terminationStatus == 0 {
            let components = schedule(job)
            var observation = BrainLayerLaunchdJobObservation(
                loadState: result.output.contains("pid =") ? .running : .loaded,
                runs: integerLine(named: "runs", in: result.output),
                lastExitCode: integerLine(named: "last exit code", in: result.output).map(Int32.init),
                lastRunAt: runRecordDate(job),
                nextRunAt: nextRun(components: components),
                isContinuous: components == nil && job.isContinuous
            )
            if job == .watch, let exit = uncleanLaunchdExit(result) {
                observation.uncleanExit = "Watcher " + exit
                observation.healthySince = launchdHealthySince(result)
            }
            return observation
        }

        if result.terminationStatus == 113 ||
            result.output.localizedCaseInsensitiveContains("could not find service") {
            return .stateOnly(.unloaded)
        }
        return .stateOnly(.probeError("launchctl exited \(result.terminationStatus)"))
    }

    private func component(_ job: BrainLayerLaunchdJob) -> BrainLayerLaunchdJobObservation {
        let name = job.rawValue
        let result = commandRunner([BrainBarSupervisorRuntime.cli, "jobs", "component", name, "status"])
        guard result.terminationStatus == 0, let runtime = try? BrainBarSupervisorRuntime.decode(result.output) else {
            return .stateOnly(.probeError("Supervisor status unavailable for \(name)"))
        }
        let state: BrainLayerLaunchdLoadState
        if runtime.paused || !runtime.configured { state = .paused }
        else if runtime.pid != nil { state = .running }
        else if let error = runtime.error, !error.isEmpty { state = .failed(error) }
        else if let code = runtime.last_exit_code, code != 0 { state = .failed("last exit \(code)") }
        else { state = .loaded }
        var observation = BrainLayerLaunchdJobObservation(
            loadState: state, runs: runtime.starts, lastExitCode: runtime.last_exit_code,
            lastRunAt: runtime.last_run_at.map(Date.init(timeIntervalSince1970:)),
            nextRunAt: runtime.next_run_at.map(Date.init(timeIntervalSince1970:)),
            isContinuous: runtime.next_run_at == nil
        )
        observation.pid = runtime.pid
        observation.ownerNote = "Supervisor component \(name)" + (runtime.pid.map { " · PID \($0)" } ?? "")
        if case .failed = state { observation.ownerNoteTone = .warning }
        return observation
    }

    private func calendarStep(_ job: BrainLayerLaunchdJob, parent: BrainLayerLaunchdCommandResult,
                              history: BrainBarCalendarHistory, next: Date?, problem: String?) -> BrainLayerLaunchdJobObservation {
        var observation = calendarStepHistory(job, parent: parent, history: history, next: next)
        observation.calendarScheduleProblem = problem
        return observation
    }

    /// S5 (R2 §5.B): the run history is read BEFORE the schedule. A parent-probe or schedule error
    /// attaches uncertainty to a known run; it only stands alone when this step has no run at all.
    private func calendarStepHistory(_ job: BrainLayerLaunchdJob, parent: BrainLayerLaunchdCommandResult,
                                     history: BrainBarCalendarHistory, next: Date?) -> BrainLayerLaunchdJobObservation {
        if let error = history.error { return .stateOnly(.probeError(error)) }
        let name = job.launchdLabel.replacingOccurrences(of: "com.brainlayer.", with: "")
        let run = history.runs[name]
        let uncertainty: String? = parent.terminationStatus != 0 ? "Maintenance calendar status unavailable"
            : next == nil ? "Maintenance calendar schedule unavailable" : nil
        var observation = calendarStepObservation(job, run: run, history: history, next: next)
        if run == nil, observation.calendarCoveredByWeeklyAt == nil, let uncertainty {
            // Nothing known about this step: unknown, never "awaiting first run".
            return .stateOnly(.probeError(uncertainty))
        }
        observation.calendarScheduleUncertainty = uncertainty
        observation.calendarConsecutiveFailures = run?.consecutiveFailures
        return observation
    }

    private func calendarStepObservation(_ job: BrainLayerLaunchdJob, run: BrainBarCalendarHistory.Run?,
                                         history: BrainBarCalendarHistory, next: Date?) -> BrainLayerLaunchdJobObservation {
        // Coverage belongs to the replacing runner's scheduled night. A later runner without a
        // nightly result must fall back to the nightly's own history, even if the weekly was ok.
        // Only journal order proves replacement; missing order must preserve the real result.
        if job == .maintenanceNightly, let weekly = history.runs["maintenance-weekly"],
           let scheduledDate = history.latestScheduledDate, !scheduledDate.isEmpty,
           weekly.scheduledDate == scheduledDate,
           weekly.replaces == "maintenance-nightly", weekly.status == "ok", weekly.code == 0,
           weekly.startedAt != nil, let weeklyOrdinal = weekly.journalOrdinal,
           run.map({ nightly in nightly.journalOrdinal.map { $0 < weeklyOrdinal } ?? false }) ?? true {
            var covered = BrainLayerLaunchdJobObservation(loadState: .loaded, runs: nil, lastExitCode: nil,
                                                          lastRunAt: nil, nextRunAt: next, isContinuous: false)
            covered.calendarCoveredByWeeklyAt = weekly.at
            covered.ownerNote = "Nightly covered by weekly"
            return covered
        }
        let state: BrainLayerLaunchdLoadState
        if let run, run.status == "failed" || run.status == "stopped" || run.status == "abandoned" || (run.code != 0 && run.code != 75) {
            state = .failed(run.status == "abandoned" ? "\(job.title): calendar run abandoned"
                : "\(job.title): calendar result \(run.status), exit \(run.code)")
        } else { state = .loaded }
        var observation = BrainLayerLaunchdJobObservation(
            loadState: state, runs: run == nil ? 0 : nil, lastExitCode: run?.status == "abandoned" ? nil : run?.code,
            lastRunAt: run?.at, nextRunAt: next, isContinuous: false
        )
        observation.calendarRunStatus = run?.status
        observation.calendarStartedAt = run?.startedAt
        if let run, state == .loaded, run.code == 75 {
            observation.calendarDisabled = run.startedAt == nil && (run.reason == "disabled" || (run.reason == nil && calendarIsDisabled(job)))
            if job == .index {
                observation.calendarIndexDeferralReason = ["resident-writer-busy": "resident writer busy",
                    "maintenance-window": "maintenance window"][run.reason ?? ""]
            }
            observation.calendarDeferralRecord = observation.calendarIndexDeferralReason == nil ? calendarAbort(job, run: run) : nil
            if job == .index, observation.calendarIndexDeferralReason == nil,
               let reason = BrainLayerMaintenanceExit.thisRunsReason(observation.calendarDeferralRecord, lastRunAt: run.startedAt) {
                observation.calendarIndexDeferralReason = ["resident-writer-busy": "resident writer busy",
                    "maintenance-window": "maintenance window"][reason]
            }
            if run.startedAt == nil, job == .maintenanceNightly || job == .maintenanceWeekly {
                switch run.reason {
                case "index-overran-window": observation.calendarMissedWindowReason = "Indexing ran past the 04:00–06:00 window"
                case "quiet-window-missed": observation.calendarMissedWindowReason = "The 04:00–06:00 maintenance window was missed"
                default: break // Old/unproven exit 75 must keep its reason-unavailable verdict.
                }
            }
        }
        observation.ownerNote = "com.brainlayer.maintenance · " + (run.map { "last result: \($0.status)" }
            ?? "Scheduled, not yet run under the supervisor")
        // A weekly that ran ok in the nightly's place keeps its own "last result: ok"; the nightly
        // row says it was covered. A weekly that did not finish ok still says the nightly was skipped.
        if job == .maintenanceWeekly, let run, run.replaces == "maintenance-nightly",
           !(run.code == 0 && run.status == "ok" && run.startedAt != nil),
           history.runs["maintenance-nightly"].map({ $0.at < run.at }) ?? true {
            observation.ownerNote = "Nightly skipped: weekly scheduled instead"
        }
        if case .failed = state { observation.ownerNoteTone = .warning }
        return observation
    }

    private func calendarAbort(_ job: BrainLayerLaunchdJob, run: BrainBarCalendarHistory.Run) -> BrainLayerMaintenanceEvidence.RunRecord? {
        guard run.startedAt != nil else { return nil } // disabled/window deferrals never spawned a child
        let name = job.launchdLabel.replacingOccurrences(of: "com.brainlayer.", with: "")
        let path = calendarLogsDirectory().appendingPathComponent("\(name).out.log")
        guard BrainBarBackupSources.isRegularFile(path),
              case let .read(data, stamp) = BrainBarStableFileReader.readTail(path, maxBytes: 64 * 1024),
              stamp <= run.at.addingTimeInterval(BrainLayerMaintenanceExit.recordClockSlack),
              let last = String(decoding: data, as: UTF8.self).split(whereSeparator: \.isNewline).last else { return nil }
        if job == .index, last.hasPrefix("INDEX_DEFERRED ") {
            let token = last.split(separator: " ").dropFirst().first.map(String.init)
            if let reason = ["resident_writer_busy": "resident-writer-busy",
                             "maintenance_window=04:00-06:00": "maintenance-window"][token ?? ""] {
                return .aborted(reason: reason, writtenAt: stamp)
            }
        }
        guard let row = try? JSONSerialization.jsonObject(with: Data(last.utf8)) as? [String: Any],
              row["status"] as? String == "aborted", let reason = row["reason"] as? String else { return nil }
        return .aborted(reason: reason, writtenAt: stamp) // verdict also checks the child's start time
    }

    private static func readCalendarLogsDirectory() -> URL {
        BrainBarBackupSources.Paths.readCalendarLogsDirectory()
    }

    private func integerLine(named key: String, in output: String) -> Int? {
        let pattern = "(?m)^\\s*\(NSRegularExpression.escapedPattern(for: key))\\s*=\\s*(-?\\d+)"
        return firstInteger(matching: pattern, in: output)
    }

    private func firstInteger(matching pattern: String, in output: String) -> Int? {
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: output, range: NSRange(output.startIndex..., in: output)),
              let range = Range(match.range(at: 1), in: output)
        else { return nil }
        return Int(output[range])
    }

    private func nextRun(components: DateComponents?) -> Date? {
        guard var components, components.hour != nil, components.minute != nil else { return nil }
        components.second = 0
        return calendar.nextDate(after: now(), matching: components, matchingPolicy: .nextTime)
    }

    static func readRunRecord(
        _ job: BrainLayerLaunchdJob,
        in directory: URL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".local/share/brainlayer/job-runs")
    ) -> Date? {
        let path = directory.appendingPathComponent("\(job.launchdLabel).started")
        guard let raw = try? String(contentsOf: path, encoding: .utf8),
              let seconds = TimeInterval(raw.trimmingCharacters(in: .whitespacesAndNewlines)) else { return nil }
        return Date(timeIntervalSince1970: seconds)
    }

    static func readSchedule(
        _ job: BrainLayerLaunchdJob,
        in directory: URL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/LaunchAgents")
    ) -> DateComponents? {
        readSchedule(label: job.launchdLabel, in: directory)
    }

    static func readSchedule(label: String, in directory: URL = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/LaunchAgents")) -> DateComponents? {
        let path = directory.appendingPathComponent("\(label).plist")
        guard let data = try? Data(contentsOf: path),
              let plist = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any],
              let interval = plist["StartCalendarInterval"] as? [String: Int],
              let hour = interval["Hour"], let minute = interval["Minute"] else { return nil }
        var components = DateComponents(hour: hour, minute: minute)
        if let weekday = interval["Weekday"] { components.weekday = (weekday % 7) + 1 }
        return components
    }

    /// Why an INSTALLED calendar LaunchAgent is not a usable schedule; nil when it is usable or absent.
    static func scheduleProblem(label: String, in directory: URL = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/LaunchAgents")) -> String? {
        let name = "\(label).plist"
        let path = directory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: path.path) else { return nil }
        guard let data = try? Data(contentsOf: path) else { return "\(name) is unreadable" }
        guard let plist = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any] else {
            return "\(name) is malformed: not a property-list dictionary"
        }
        guard let interval = plist["StartCalendarInterval"] as? [String: Int],
              interval["Hour"] != nil, interval["Minute"] != nil else {
            return "\(name) is malformed: no StartCalendarInterval Hour and Minute"
        }
        return nil
    }

    private static func run(_ command: [String]) -> BrainLayerLaunchdCommandResult {
        let result = LaunchctlWatcherProcessProbe.run(command,
            timeout: command.first == BrainBarSupervisorRuntime.cli ? 5 : 1)
        return BrainLayerLaunchdCommandResult(terminationStatus: result.terminationStatus, output: result.output)
    }

}

private extension BrainLayerLaunchdJob {
    var isContinuous: Bool {
        switch self {
        case .watch, .drain, .hotlane: true
        default: false
        }
    }
}

struct StaticBrainLayerLaunchdStatusProvider: BrainLayerLaunchdStatusSampling {
    let states: [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState]

    func sample() -> [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] {
        states
    }
}
