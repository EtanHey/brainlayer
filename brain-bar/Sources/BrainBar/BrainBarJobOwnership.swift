import Darwin
import Dispatch
import Foundation

// Single vocabulary for maintenance presentation; matches night-recovery (#1550).
enum BrainBarMaintenanceStep {
    static let labels = ["jsonl-backup": "Transcript backup", "backup-daily": "Database backup",
        "backup-snapshot": "Database snapshot", "index": "Transcript indexing",
        "maintenance-nightly": "Nightly maintenance", "maintenance-weekly": "Weekly maintenance",
        "wal-checkpoint": "Database checkpoint", "repair-fts": "Search index repair",
        "decay": "Memory decay", "p0-counter": "Memory counts"]

    static func label(_ step: String) -> String { labels[step] ?? "Maintenance step" }
}

enum BrainBarJobOwnership: Equatable, Sendable {
    case legacy, supervisor, unknown(String)

    static let directory = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".local/share/brainlayer/supervisor")

    static func read(
        journalURL: URL = directory.appendingPathComponent("migration.json"),
        daemonProbe: () -> Int32? = { probeDaemon() }
    ) -> Self {
        do {
            // Rollback payloads, published plists and the manifest make the live journal >100 KB.
            // 4 MiB leaves room for that inventory while keeping the private read finite.
            guard let data = try readOwnedFile(journalURL, privateMode: true, limit: 4 * 1024 * 1024) else {
                let manifest = journalURL.deletingLastPathComponent().appendingPathComponent("jobs.json")
                if try readOwnedFile(manifest, privateMode: true, limit: 4 * 1024 * 1024) != nil {
                    return .unknown("Can't determine job owner: migration journal missing but supervisor manifest exists")
                }
                let daemon = daemonProbe()
                guard daemon == 113 else {
                    return .unknown("Can't determine job owner: migration journal missing and supervisor daemon is loaded or its absence cannot be proven")
                }
                return .legacy
            }
            guard let row = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let phase = row["phase"] as? String else { throw CocoaError(.fileReadCorruptFile) }
            switch phase {
            case "migrating", "migrated", "reverting": return .supervisor
            case "reverted", "rolled_back": return .legacy
            default: throw CocoaError(.fileReadCorruptFile)
            }
        } catch {
            return .unknown("Can't determine job owner: migration journal is unreadable or untrusted")
        }
    }

    private static func probeDaemon() -> Int32? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["print", "gui/\(getuid())/com.brainlayer.daemon"]
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        let finished = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in finished.signal() }
        do { try process.run() } catch { return nil }
        guard finished.wait(timeout: .now() + 1) == .success else {
            if process.isRunning { process.terminate() }
            return nil
        }
        guard process.terminationReason == .exit else { return nil }
        return process.terminationStatus
    }

    // Same custody checks as install.sh's journal guard. Open every path segment with NOFOLLOW:
    // a parent link must not turn an apparently private journal into a different owner's file.
    static func readOwnedFile(_ url: URL, privateMode: Bool, limit: Int) throws -> Data? {
        var fd = Darwin.open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard fd >= 0 else { throw CocoaError(.fileReadNoPermission) }
        defer { Darwin.close(fd) }
        let parts = url.pathComponents.dropFirst()
        guard !parts.contains(".."), !parts.contains(".") else { throw CocoaError(.fileReadNoPermission) }
        for (index, part) in parts.enumerated() {
            let last = index == parts.count - 1
            let next = openat(fd, part, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC | (last ? 0 : O_DIRECTORY))
            if next < 0 {
                if errno == ENOENT { return nil }
                throw CocoaError(.fileReadNoPermission)
            }
            Darwin.close(fd)
            fd = next
        }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG,
              info.st_uid == getuid(), info.st_nlink == 1,
              info.st_mode & (privateMode ? 0o077 : 0o022) == 0,
              info.st_size <= limit else { throw CocoaError(.fileReadNoPermission) }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 8192)
        while true {
            let count = Darwin.read(fd, &buffer, buffer.count)
            guard count >= 0 else { throw CocoaError(.fileReadUnknown) }
            if count == 0 { return data }
            guard data.count + count <= limit else { throw CocoaError(.fileReadTooLarge) }
            data.append(contentsOf: buffer.prefix(count))
        }
    }
}

struct BrainBarSupervisorRuntime: Decodable, Sendable {
    let configured: Bool
    let paused: Bool
    let pid: Int32?
    let starts: Int
    let failures: Int
    let last_exit_code: Int32?
    let last_run_at: Double?
    let next_run_at: Double?
    let error: String?

    static let cli = "/opt/homebrew/opt/brainlayer/bin/brainlayer"

    static func decode(_ output: String) throws -> Self {
        struct Response: Decodable { let source: String; let runtime: BrainBarSupervisorRuntime }
        let response = try JSONDecoder().decode(Response.self, from: Data(output.utf8))
        let value = response.runtime
        guard response.source == "supervisor", value.starts >= 0, value.failures >= 0,
              value.pid == nil || value.pid! > 1,
              [value.last_run_at, value.next_run_at].compactMap({ $0 }).allSatisfy({ $0.isFinite && $0 >= 0 })
        else { throw CocoaError(.fileReadCorruptFile) }
        return value
    }
}

struct BrainBarCalendarHistory: Sendable {
    struct StepProblem: Equatable, Sendable {
        let at: Date
        var cutOff = false
    }
    struct NightOutcome: Equatable, Sendable {
        let scheduledDate: String
        let at: Date
        var failed: Bool
        var abandoned: Bool
        var consecutiveFailures = 0
        var failedSteps: [String] = []
        var stepProblems: [String: StepProblem] = [:]
        var runnerErrorAt: Date? = nil
        var unknownFailureAt: Date? = nil
        /// An abandonment that named no step: its own cause, separate from later unknown failures.
        var abandonmentFailureAt: Date? = nil
    }
    private enum AttemptStep {
        case succeeded, failed, incomplete, deferred
        var isProblem: Bool { self == .failed || self == .incomplete }
    }
    /// Nightly terminals earn the dot; later same-date step completions can resolve it.
    var maintenanceNight: NightOutcome? = nil
    struct Run: Sendable {
        let at: Date
        let code: Int32
        let status: String
        var startedAt: Date? = nil
        var reason: String? = nil
        var replaces: String? = nil
        var scheduledDate: String? = nil
        // Position in this journal snapshot; wall-clock timestamps can move backwards.
        var journalOrdinal: Int? = nil
        /// Failed runs of this step in a row, ending with this one (S5).
        var consecutiveFailures = 0
    }
    var runs: [String: Run] = [:]
    var error: String? = nil
    var latestScheduledDate: String? = nil
    static let empty = Self()

    static func read(directory: URL = BrainBarJobOwnership.directory.appendingPathComponent("calendar")) -> Self {
        do {
            guard let data = try BrainBarJobOwnership.readOwnedFile(
                directory.appendingPathComponent("runs.jsonl"), privateMode: false, limit: 4 * 1024 * 1024
            ) else { return .empty }
            var result = Self()
            var starts: [String: Date] = [:]
            var streaks: [String: Int] = [:]
            var abandonedRuns: Set<String> = []
            var nights: [String: NightOutcome] = [:]
            var unresolvedSteps: [String: Set<String>] = [:]
            var stepProblems: [String: [String: StepProblem]] = [:]
            var unknownFailures: [String: Date] = [:]
            var abandonmentFailures: [String: Date] = [:]
            var runnerErrors: [String: Date] = [:]
            var abandonmentSummaries: Set<String> = []
            var openAttempt: String? = nil
            var pendingRunnerError: Date? = nil
            // Finished rows inherit their date from the run header. Consume each attempt at
            // its terminal row so pre-failure successes cannot prove a later recovery.
            var attemptSteps: [String: [String: AttemptStep]] = [:]
            var attemptKinds: [String: String] = [:]
            for (ordinal, line) in String(decoding: data, as: UTF8.self).split(whereSeparator: \.isNewline).enumerated() {
                guard let row = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any]
                else { throw CocoaError(.fileReadCorruptFile) }
                let event = row["event"] as? String
                if event == "runner" || event == "transcript-recovery" {
                    starts.removeAll()
                    result.latestScheduledDate = row["scheduled_date"] as? String
                    if let date = result.latestScheduledDate {
                        if attemptKinds[date] == "nightly" {
                            unresolvedSteps[date, default: []].formUnion((attemptSteps[date] ?? [:]).filter { $0.value.isProblem }.map(\.key))
                        }
                        attemptSteps[date] = [:]
                        attemptKinds[date] = row["kind"] as? String ?? (event == "runner" ? "nightly" : "recovery")
                        abandonmentSummaries.remove(date)
                    }
                    openAttempt = result.latestScheduledDate
                    pendingRunnerError = nil
                }
                if event == "runner-error" {
                    guard let ts = row["ts"] as? Double, ts.isFinite else { throw CocoaError(.fileReadCorruptFile) }
                    let at = Date(timeIntervalSince1970: ts)
                    // Older producers left the row undated. Inside an open attempt it belongs to that
                    // attempt; before any header it failed pre-launch, and its own summary names it.
                    if let date = row["scheduled_date"] as? String ?? openAttempt {
                        if (row["kind"] as? String ?? attemptKinds[date]) == "nightly", runnerErrors[date] == nil { runnerErrors[date] = at }
                    } else if pendingRunnerError == nil {
                        pendingRunnerError = at
                    }
                }
                if event == "summary" {
                    if let date = row["scheduled_date"] as? String, let at = pendingRunnerError,
                       row["kind"] as? String == "nightly", runnerErrors[date] == nil {
                        runnerErrors[date] = at
                    }
                    openAttempt = nil
                    pendingRunnerError = nil
                }
                let nightly = row["kind"] as? String == "nightly"
                if (nightly && (event == "summary" || event == "abandoned")) ||
                    (event == "summary" && row["kind"] is String &&
                        (row["scheduled_date"] as? String).flatMap({ nights[$0] }) != nil) {
                    guard let date = row["scheduled_date"] as? String,
                          date.range(of: #"^\d{4}-\d{2}-\d{2}$"#, options: .regularExpression) != nil,
                          let ts = row["ts"] as? Double, ts.isFinite else { throw CocoaError(.fileReadCorruptFile) }
                    let abandoned = event == "abandoned"
                    let failures = row["failed"] as? Int
                    guard abandoned || failures.map({ $0 >= 0 }) == true else { throw CocoaError(.fileReadCorruptFile) }
                    let previous = nights[date]
                    // A late abandonment can describe an older nightly attempt while the
                    // active header belongs to hourly recovery. Only its nightly evidence adds problems.
                    let steps = nightly && attemptKinds[date] != "nightly" ? [:] : attemptSteps[date] ?? [:]
                    // An abandonment is followed by a synthetic failed=1 summary, not a
                    // new runner failure. Consume finished evidence only once, at either terminal.
                    let abandonmentSummary = event == "summary" &&
                        (row["abandoned"] as? Bool == true || abandonmentSummaries.contains(date))
                    if abandoned { abandonmentSummaries.insert(date) }
                    attemptSteps.removeValue(forKey: date)
                    if event == "summary" {
                        abandonmentSummaries.remove(date)
                    }
                    var unresolved = unresolvedSteps[date] ?? []
                    unresolved.subtract(steps.filter { $0.value == .succeeded }.map(\.key))
                    // Recovery/manual attempts can only resolve existing nightly problems.
                    if nightly && (abandoned || (failures ?? 0) > 0) {
                        unresolved.formUnion(steps.filter { $0.value.isProblem }.map(\.key))
                        if let incomplete = row["incomplete_steps"] {
                            guard let names = incomplete as? [String] else { throw CocoaError(.fileReadCorruptFile) }
                            unresolved.formUnion(names)
                            for step in names {
                                stepProblems[date, default: [:]][step] = StepProblem(at: Date(timeIntervalSince1970: ts), cutOff: true)
                            }
                        }
                        // Only failed finished rows in THIS attempt explain the summary count.
                        // Deferred/skipped/stopped steps and inherited problems cannot absorb
                        // a runner failure. Neither kind of non-step evidence is recoverable
                        // through later step successes.
                        let failedHere = steps.values.filter { $0 == .failed }.count
                        // An incomplete nightly plan reports at least one failure on its own
                        // (failed=max(1, ...)), so a signal stop is explained by its cut-off steps.
                        let planIncomplete = steps.values.contains(.incomplete) ||
                            (row["incomplete_steps"] as? [String])?.isEmpty == false
                        let expected = max(planIncomplete ? 1 : 0, failedHere)
                        if unresolved.isEmpty && (abandoned || abandonmentSummary) {
                            // The abandonment itself is the cause; a later attempt's failure stays separate.
                            if abandonmentFailures[date] == nil { abandonmentFailures[date] = Date(timeIntervalSince1970: ts) }
                        } else if unresolved.isEmpty || (!abandoned && !abandonmentSummary && (failures ?? 0) > expected) {
                            if unknownFailures[date] == nil { unknownFailures[date] = Date(timeIntervalSince1970: ts) }
                        }
                    }
                    unresolvedSteps[date] = unresolved
                    stepProblems[date] = (stepProblems[date] ?? [:]).filter { unresolved.contains($0.key) }
                    nights[date] = NightOutcome(scheduledDate: date,
                        at: nightly ? Date(timeIntervalSince1970: ts) : (previous?.at ?? Date(timeIntervalSince1970: ts)),
                        failed: !unresolved.isEmpty || unknownFailures[date] != nil || abandonmentFailures[date] != nil
                            || runnerErrors[date] != nil,
                        abandoned: abandoned || previous?.abandoned == true)
                }
                if row["event"] as? String == "spawn", let step = row["step"] as? String,
                   let ts = row["ts"] as? Double, ts.isFinite {
                    starts[step] = Date(timeIntervalSince1970: ts)
                }
                if row["event"] as? String == "abandoned" {
                    guard let date = row["scheduled_date"] as? String,
                          let ts = row["ts"] as? Double, ts.isFinite else { throw CocoaError(.fileReadCorruptFile) }
                    guard row["kind"] as? String == "nightly", abandonedRuns.insert(date).inserted else { continue }
                    // Step projections are display evidence only; the dot uses the nightly outcomes.
                    let incomplete = row["incomplete_steps"] as? [String] ?? ["maintenance-nightly", "maintenance-weekly"]
                    for step in incomplete where ["maintenance-nightly", "maintenance-weekly"].contains(step) {
                        guard result.runs[step]?.scheduledDate != date else { continue }
                        if step == "maintenance-nightly", let weekly = result.runs["maintenance-weekly"],
                           weekly.scheduledDate == date, weekly.status == "ok", weekly.code == 0,
                           weekly.replaces == "maintenance-nightly" { continue }
                        streaks[step, default: 0] += 1
                        result.runs[step] = Run(at: Date(timeIntervalSince1970: ts), code: -1, status: "abandoned",
                            startedAt: (row["runner_started_at"] as? Double).map(Date.init(timeIntervalSince1970:)),
                            reason: row["reason"] as? String, scheduledDate: date, journalOrdinal: ordinal,
                            consecutiveFailures: streaks[step] ?? 0)
                    }
                    result.latestScheduledDate = date
                    continue
                }
                guard row["event"] as? String == "finished" else { continue }
                guard let step = row["step"] as? String, let ts = row["ts"] as? Double, ts.isFinite,
                      let status = row["status"] as? String else { throw CocoaError(.fileReadCorruptFile) }
                // Shutdown records the unstarted remainder as skipped/null. It did not run,
                // so preserve the step's preceding result rather than invalidating all history.
                let startedAt = starts.removeValue(forKey: step)
                let scheduledDate = row["scheduled_date"] as? String ?? result.latestScheduledDate
                if status == "skipped", row["returncode"] is NSNull {
                    if let date = scheduledDate {
                        attemptSteps[date, default: [:]][step] = .incomplete
                        if attemptKinds[date] == "nightly", stepProblems[date]?[step] == nil {
                            stepProblems[date, default: [:]][step] = StepProblem(at: Date(timeIntervalSince1970: ts), cutOff: true)
                        }
                    }
                    continue
                }
                guard let code = row["returncode"] as? Int32,
                      ["ok", "failed", "deferred", "stopped"].contains(status)
                else { throw CocoaError(.fileReadCorruptFile) }
                if let date = scheduledDate {
                    let outcome: AttemptStep = status == "deferred" || code == 75 ? .deferred :
                        status == "ok" && code == 0 ? .succeeded : status == "failed" ? .failed : .incomplete
                    attemptSteps[date, default: [:]][step] = outcome
                    // A retry that never ran cannot replace the original problem's time.
                    if attemptKinds[date] == "nightly", outcome.isProblem {
                        stepProblems[date, default: [:]][step] = StepProblem(at: Date(timeIntervalSince1970: ts), cutOff: status == "stopped")
                    }
                }
                // A deferral (exit 75) did not run the step: it neither extends nor resets the streak.
                if status == "failed" || status == "stopped" || (code != 0 && code != 75) {
                    streaks[step, default: 0] += 1
                } else if status == "ok" {
                    streaks[step] = 0
                }
                result.runs[step] = Run(at: Date(timeIntervalSince1970: ts), code: code, status: status,
                    startedAt: startedAt, reason: row["reason"] as? String, replaces: row["replaces"] as? String,
                    scheduledDate: row["scheduled_date"] as? String ?? result.latestScheduledDate, journalOrdinal: ordinal,
                    consecutiveFailures: streaks[step] ?? 0)
            }
            var streak = 0
            for date in nights.keys.sorted() {
                guard var night = nights[date] else { continue }
                night.runnerErrorAt = runnerErrors[date]
                night.unknownFailureAt = unknownFailures[date]
                night.abandonmentFailureAt = abandonmentFailures[date]
                night.failed = night.failed || night.runnerErrorAt != nil || night.unknownFailureAt != nil
                    || night.abandonmentFailureAt != nil
                streak = night.failed ? streak + 1 : 0
                night.consecutiveFailures = streak
                night.failedSteps = (unresolvedSteps[date] ?? []).sorted()
                night.stepProblems = stepProblems[date] ?? [:]
                result.maintenanceNight = night
            }
            return result
        } catch { return Self(error: "Maintenance calendar history is unreadable") }
    }
}
