import Foundation
import SwiftUI

struct BrainBarMaintenanceRetryReceipt: Decodable, Sendable {
    struct Failure: Decodable, Sendable { let step: String; let reason: String }
    let status: String
    let reason: String
    var steps: [String] = []
    var failures: [Failure] = []
    var deferred: [Failure] = []
    let retryPauseStart: String?
    let retryPauseEnd: String?
    /// The scheduled night this receipt describes.
    let scheduledDate: String?

    enum CodingKeys: String, CodingKey {
        case status, reason, steps, failures, deferred
        case retryPauseStart = "retry_pause_start", retryPauseEnd = "retry_pause_end"
        case scheduledDate = "scheduled_date"
    }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        status = try values.decode(String.self, forKey: .status)
        reason = try values.decode(String.self, forKey: .reason)
        steps = try values.decodeIfPresent([String].self, forKey: .steps) ?? []
        failures = try values.decodeIfPresent([Failure].self, forKey: .failures) ?? []
        deferred = try values.decodeIfPresent([Failure].self, forKey: .deferred) ?? []
        retryPauseStart = try values.decodeIfPresent(String.self, forKey: .retryPauseStart)
        retryPauseEnd = try values.decodeIfPresent(String.self, forKey: .retryPauseEnd)
        scheduledDate = try values.decodeIfPresent(String.self, forKey: .scheduledDate)
    }
}

@MainActor
final class BrainBarMaintenanceRetryModel: ObservableObject {
    @Published private(set) var receipt: BrainBarMaintenanceRetryReceipt?
    @Published private(set) var isRunning = false
    @Published private(set) var outcome: String?
    /// The night `outcome` was produced for; a receipt for another night clears it.
    private var outcomeNight: String?
    @Published private(set) var error = "Checking retry availability…"
    private var runner: (any BrainLayerCLIRunning)?
    private let now: () -> Date
    private let calendar: Calendar
    private var refreshTask: Task<Void, Never>?
    private var generation = 0

    init(runner: (any BrainLayerCLIRunning)?, now: @escaping () -> Date = Date.init, calendar: Calendar = .current) {
        self.runner = runner
        self.now = now
        self.calendar = calendar
    }

    func install(runner: any BrainLayerCLIRunning) { self.runner = runner }

    private var pauseInterval: DateInterval? {
        let formatter = ISO8601DateFormatter()
        guard let start = receipt?.retryPauseStart.flatMap({ formatter.date(from: $0) }),
              let end = receipt?.retryPauseEnd.flatMap({ formatter.date(from: $0) }), end > start else { return nil }
        return DateInterval(start: start, end: end)
    }
    private var inWindow: Bool {
        guard let interval = pauseInterval else { return false }
        return now() >= interval.start && now() < interval.end
    }
    var canRetry: Bool {
        !isRunning && !inWindow && receipt?.status == "ready" && receipt?.steps.isEmpty == false
            && pauseInterval.map { now() < $0.end } == true
    }
    /// The button exists only when pressing it would do something; never a disabled control.
    var showsRetryButton: Bool { canRetry }
    var showsProgress: Bool { isRunning }
    private var refusedOrFailed: Bool { ["refused", "failed"].contains(receipt?.status) }

    private func localTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.dateFormat = "HH:mm"
        return formatter.string(from: date)
    }

    var explanation: String {
        if isRunning { return "Retrying failed steps… This retry does not pause services." }
        if inWindow, let interval = pauseInterval {
            return "Retries are paused for scheduled maintenance (until \(localTime(interval.end)))."
        }
        // Deferred-only work is explained in `details`, unless the retry was refused or failed:
        // then the reason must still show.
        if receipt?.steps.isEmpty == true, receipt?.deferred.isEmpty == false, !refusedOrFailed { return "" }
        if receipt?.status == "ready" {
            if receipt?.steps.isEmpty == true { return "Nothing to retry now." }
            guard pauseInterval.map({ now() < $0.end }) == true else { return "Checking retry availability…" }
            let labels = (receipt?.steps ?? []).map(BrainBarMaintenanceStep.label).joined(separator: ", ")
            return "Retry will run: \(labels). Nothing else pauses."
        }
        if receipt?.status == "nothing-to-retry" { return "Everything from last night has finished. Nothing to retry." }
        if receipt?.status == "accepted" { return "The retry finished." }
        return receipt?.reason ?? error
    }
    var details: String {
        guard let receipt else { return "" }
        let deferredSteps = Set(receipt.deferred.map(\.step))
        // Deferred work is visible even when the receipt omits it from failures.
        let steps = receipt.failures.map(\.step) + receipt.deferred.map(\.step).filter { step in
            !receipt.failures.contains { $0.step == step }
        }
        return steps.map { step in
            let unfinished = "\(BrainBarMaintenanceStep.label(step)) didn’t finish last night."
            if deferredSteps.contains(step), !inWindow {
                let start = receipt.retryPauseStart.flatMap { ISO8601DateFormatter().date(from: $0) }
                let time = start.flatMap { $0 > now() ? ", after \(localTime($0))" : nil } ?? ""
                let next = " It runs again automatically tonight during scheduled maintenance\(time)."
                return unfinished + next + (showsRetryButton || showsProgress || refusedOrFailed ? "" : " Nothing to do now.")
            }
            return unfinished + (canRetry && receipt.steps.contains(step) ? " You can retry it now." : "")
        }.joined(separator: "\n")
    }

    // ProcessBrainLayerCLIRunner blocks; keep the long call off the cooperative pool.
    nonisolated private static func run(_ runner: any BrainLayerCLIRunning, arguments: [String], timeout: TimeInterval) async -> BrainLayerCLIResult? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                continuation.resume(returning: runner.run(arguments, timeout: timeout))
            }
        }
    }

    private func parse(_ result: BrainLayerCLIResult?) -> BrainBarMaintenanceRetryReceipt? {
        guard let result, !result.timedOut,
              let data = result.stdout.split(whereSeparator: \.isNewline).last.map({ Data($0.utf8) }),
              let value = try? JSONDecoder().decode(BrainBarMaintenanceRetryReceipt.self, from: data),
              ["ready", "accepted", "refused", "nothing-to-retry", "failed"].contains(value.status),
              result.terminationStatus == 0 || ["failed", "refused"].contains(value.status) else { return nil }
        return value
    }

    func refresh() async {
        if let task = refreshTask { await task.value; return }
        guard !isRunning else { return }
        guard let runner else { error = "Retry unavailable: brainlayer CLI is not configured"; return }
        let sampledGeneration = generation
        let task = Task { @MainActor in
            let result = await Self.run(runner, arguments: ["maintenance-calendar", "--recover-failed", "--status"], timeout: 30)
            guard sampledGeneration == generation else { return }
            receipt = parse(result)
            if outcome != nil, let night = receipt?.scheduledDate, night != outcomeNight { outcome = nil }
            if receipt == nil { error = "Retry availability could not be checked; see the calendar log." }
        }
        refreshTask = task
        await task.value
        refreshTask = nil
    }

    func retry() async {
        // No CLI, or a retry already running: do not launch another action.
        guard let runner, !isRunning else { return }
        guard canRetry else {
            // Eligibility was lost while the confirmation was open; say why instead of doing nothing.
            outcome = "Refused: \(explanation)"
            outcomeNight = receipt?.scheduledDate
            return
        }
        isRunning = true
        if let task = refreshTask { await task.value }
        isRunning = false
        let night = receipt?.scheduledDate
        guard canRetry else {
            outcome = "Refused: \(explanation)"
            outcomeNight = night
            return
        }
        generation += 1
        isRunning = true
        outcome = nil
        let selected = receipt?.steps ?? []
        let result = await Self.run(runner, arguments: ["maintenance-calendar", "--recover-failed"], timeout: 8_000)
        outcomeNight = night
        if var value = parse(result) {
            outcomeNight = value.scheduledDate ?? night
            if value.failures.isEmpty { value.failures = receipt?.failures ?? [] }
            if value.deferred.isEmpty { value.deferred = receipt?.deferred ?? [] }
            if value.status == "accepted" {
                value.failures.removeAll { selected.contains($0.step) }
            }
            receipt = value
            switch value.status {
            case "accepted":
                let labels = selected.map(BrainBarMaintenanceStep.label).joined(separator: ", ")
                outcome = "Done: \(labels) retried successfully."
            case "refused": outcome = "Refused: \(value.reason)"
            case "nothing-to-retry": outcome = "Nothing to retry: \(value.reason)"
            default: outcome = "Failed: \(value.reason)"
            }
        } else {
            outcome = "Failed: retry returned no supported result. Check the calendar log before trying again."
        }
        isRunning = false
        await refresh()
    }

#if DEBUG
    func setForPreview(json: String, isRunning: Bool = false, outcome: String? = nil) {
        receipt = parse(.init(terminationStatus: 0, stdout: json))
        self.isRunning = isRunning
        self.outcome = outcome
    }
#endif
}

private struct BrainBarMaintenanceRetryKey: EnvironmentKey {
    static let defaultValue: BrainBarMaintenanceRetryModel? = nil
}
extension EnvironmentValues {
    var brainBarMaintenanceRetry: BrainBarMaintenanceRetryModel? {
        get { self[BrainBarMaintenanceRetryKey.self] }
        set { self[BrainBarMaintenanceRetryKey.self] = newValue }
    }
}

struct BrainBarMaintenanceRetryRow: View {
    @ObservedObject var model: BrainBarMaintenanceRetryModel
    @State private var confirming = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !model.details.isEmpty {
                Text(model.details).fixedSize(horizontal: false, vertical: true)
            }
            if model.showsRetryButton {
                Button("Retry failed steps") { confirming = true }
                    .accessibilityIdentifier("brainbar.maintenance.retry-failed")
            } else if model.showsProgress {
                ProgressView().controlSize(.small)
            }
            if !model.explanation.isEmpty { Text(model.explanation).foregroundStyle(.secondary) }
            if let outcome = model.outcome { Text(outcome) }
        }
        .font(.system(size: 12))
        .confirmationDialog("Retry failed steps?", isPresented: $confirming) {
            Button("Retry failed steps") { Task { await model.retry() } }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Retry only failed or incomplete daytime-safe steps from the latest scheduled night. No services will pause. Steps that must wait will run in tonight's scheduled maintenance.")
        }
    }
}
