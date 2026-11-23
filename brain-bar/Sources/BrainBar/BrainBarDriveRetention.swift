import SwiftUI

/// Read-only producer evidence. No Drive client or configuration writer belongs to this surface.
struct BrainBarDriveRetention: Codable, Sendable {
    struct Rule: Codable, Sendable {
        let dailyKeep, monthlyKeep: Int
        let perMachine: Bool
        let description: String?
        var isSupported: Bool { perMachine && dailyKeep >= 1 && monthlyKeep >= 0 }
    }
    struct Summary: Codable, Sendable {
        let count: Int
        let bytes: Int64
        let oldest, newest: String?
        var isValid: Bool { count >= 0 && bytes >= 0 }
    }
    struct Representative: Codable, Sendable {
        let name, date: String
        let bytes: Int64
    }
    struct Machine: Codable, Sendable {
        let machineId, state: String
        let selectedAt, error: String?
        let onDrive, kept, wouldRemove: Summary?
        /// Optional cohorts inside `kept` (kept = kept_by_rule + unverified + pinned); none is ever added on top.
        let keptByRule, unverified, pinned: Summary?
        let namedOneOffsCount: Int?
        let monthlyRepresentatives: [Representative]?
        var measured: Bool { state == "measured" && error == nil }
    }
    let rule: Rule?
    let writesEnabled: Bool
    let configError: String?
    let machines: [Machine]

    static func from(_ result: ObservabilityReadResult) -> Self? {
        guard case let .readable(document) = result else { return nil }
        return document.backups.driveRetention
    }
}

#if BRAINBAR_UI
struct BrainBarDriveRetentionPresentation {
    typealias Say = BrainBarMenuPresenter.DriveRetentionSay
    let evidence: BrainBarDriveRetention?
    var rule: BrainBarDriveRetention.Rule? {
        guard evidence?.configError == nil, let rule = evidence?.rule, rule.isSupported else { return nil }
        return rule
    }
    var atRest: String {
        if let error = evidence?.configError { return Say.configError(error) }
        guard let rule else { return Say.notReported }
        return Say.atRest(rule.dailyKeep, rule.monthlyKeep)
    }
    /// The producer currently offers no writable path, even if an unexpected payload says writesEnabled=true.
    var canEnableCleanup: Bool { false }
    var total: (count: Int, bytes: Int64)? {
        guard rule != nil, let machines = evidence?.machines, !machines.isEmpty else { return nil }
        var count = 0, bytes: Int64 = 0
        for machine in machines {
            guard machine.measured, let summary = machine.wouldRemove, summary.isValid else { return nil }
            let nextCount = count.addingReportingOverflow(summary.count)
            let nextBytes = bytes.addingReportingOverflow(summary.bytes)
            guard !nextCount.overflow, !nextBytes.overflow else { return nil }
            count = nextCount.partialValue; bytes = nextBytes.partialValue
        }
        return (count, bytes)
    }
    var firstCleanup: String {
        total.map { Say.firstCleanup($0.count, Self.size($0.bytes)) } ?? Say.notMeasured
    }
    static func size(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .decimal)
    }
    static func date(_ value: String, relativeTo now: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        let parsed: Date?
        if value.contains("T") {
            let iso = ISO8601DateFormatter()
            iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            parsed = iso.date(from: value) ?? ISO8601DateFormatter().date(from: value)
        } else {
            formatter.dateFormat = "yyyy-MM-dd"
            parsed = formatter.date(from: value)
        }
        guard let date = parsed else { return value }
        let cutoff = Calendar.current.date(byAdding: .month, value: -12, to: now) ?? now
        formatter.dateFormat = date >= cutoff && date <= now ? "MMM d" : "MMM d, yyyy"
        return formatter.string(from: date)
    }
    static func summary(_ value: BrainBarDriveRetention.Summary?, includingRange: Bool = false, relativeTo now: Date = Date()) -> String {
        guard let value, value.isValid else { return Say.notMeasured }
        let range = includingRange ? value.oldest.flatMap { oldest in value.newest.map { Say.dateRange(date(oldest, relativeTo: now), date($0, relativeTo: now)) } } : nil
        return Say.summary(value.count, size(value.bytes)) + (range.map { "; " + $0 } ?? "")
    }
    static func keeps(_ machine: BrainBarDriveRetention.Machine, relativeTo now: Date = Date()) -> String {
        guard let value = machine.keptByRule ?? machine.kept, value.isValid else { return Say.keepsNotMeasured }
        return Say.keeps(value.count, size(value.bytes), machine.keptByRule == nil ? nil : value.newest.map { date($0, relativeTo: now) })
    }
    static func status(_ machine: BrainBarDriveRetention.Machine, relativeTo now: Date = Date()) -> String {
        switch machine.state {
        case "stale": return Say.stale(machine.selectedAt.map { date($0, relativeTo: now) } ?? Say.notMeasured, machine.error)
        default: return machine.error.map(Say.couldNotMeasure) ?? Say.notMeasured
        }
    }
}

/// Both Settings designs use the same button and sheet, including the missing-evidence state.
struct BrainBarDriveRetentionLink: View {
    let evidence: BrainBarDriveRetention?
    var classic = false
    @State private var showing = false
    private typealias Say = BrainBarMenuPresenter.DriveRetentionSay
    var body: some View {
        Button { showing = true } label: {
            HStack(spacing: 5) {
                Text(BrainBarDriveRetentionPresentation(evidence: evidence).atRest)
                    .fixedSize(horizontal: false, vertical: true)
                Image(systemName: "info.circle")
            }
            .font(NDType.caption)
            .foregroundStyle(evidence?.configError != nil
                ? (classic ? Color(nsColor: BrainBarDesignTokens.Colors.statusAttention) : ND.orangeText)
                : (classic ? Color.brainBarTextSecondary : ND.t2))
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(Say.showDetails)
        .accessibilityIdentifier("brainbar.drive-retention.details")
        .sheet(isPresented: $showing) {
            BrainBarDriveRetentionSheet(evidence: evidence) { showing = false }
        }
    }
}

struct BrainBarDriveRetentionSheet: View {
    let evidence: BrainBarDriveRetention?
    var referenceDate = Date()
    var done: () -> Void = {}
    @State private var confirming = false
    private typealias Say = BrainBarMenuPresenter.DriveRetentionSay
    private var presentation: BrainBarDriveRetentionPresentation { .init(evidence: evidence) }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text(Say.title).font(NDType.bodyStrong)
                Text(Say.purpose).font(NDType.caption).foregroundStyle(ND.t2)
            }
            section(Say.rule) {
                if let rule = presentation.rule {
                    valueRow(Say.daily, String(rule.dailyKeep))
                    ND.hair.frame(height: 0.5)
                    valueRow(Say.monthly, String(rule.monthlyKeep))
                } else {
                    caption(evidence?.configError.map(Say.configError) ?? Say.notReported)
                }
            }
            if let rule = presentation.rule {
                Text(Say.explanation(rule.dailyKeep, rule.monthlyKeep))
                    .fixedSize(horizontal: false, vertical: true)
                    .font(NDType.caption).foregroundStyle(ND.t3).padding(.horizontal, 12)
            }
            section(Say.onDrive) {
                if let machines = evidence?.machines, !machines.isEmpty {
                    ViewThatFits(in: .vertical) {
                        machineList(machines).fixedSize(horizontal: false, vertical: true)
                        ScrollView { machineList(machines) }
                    }
                    .frame(maxHeight: 280)
                    .fixedSize(horizontal: false, vertical: true)
                } else { caption(Say.notMeasured) }
            }
            if evidence?.machines.isEmpty == false {
                Text(presentation.firstCleanup).font(NDType.caption).foregroundStyle(ND.t3).padding(.horizontal, 12)
            }
            section(Say.cleanup) {
                VStack(alignment: .leading, spacing: 8) {
                    Toggle(isOn: Binding(get: { false }, set: { requested in
                        if requested && presentation.canEnableCleanup { confirming = true }
                    })) { Text(Say.removeOld).frame(maxWidth: .infinity, alignment: .leading) }
                    .toggleStyle(.switch).controlSize(.mini)
                    .disabled(!presentation.canEnableCleanup)
                    .accessibilityIdentifier("brainbar.drive-retention.cleanup")
                    Text(Say.cleanupOff).font(NDType.caption).foregroundStyle(ND.t3)
                    Text(Say.lastCleanupNotReported).font(NDType.caption).foregroundStyle(ND.t3)
                    if confirming { BrainBarDriveRetentionConfirmation(presentation: presentation) { confirming = false } }
                }.padding(12)
            }
            HStack { Spacer(); NDButton(label: Say.done, small: true, action: done).keyboardShortcut(.cancelAction) }
        }
        .padding(20).frame(width: 452)
        .background(ND.bg).foregroundStyle(ND.t1)
        .environment(\.ndPalette, .window)
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(NDType.bodyStrong).foregroundStyle(ND.t2)
            VStack(alignment: .leading, spacing: 0, content: content)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 10).fill(ND.raised))
                .ndDecoration(RoundedRectangle(cornerRadius: 10).strokeBorder(ND.hair2, lineWidth: 0.5))
        }
    }
    private func valueRow(_ label: String, _ value: String) -> some View {
        HStack { Text(label); Spacer(); Text(value).foregroundStyle(ND.t2) }
            .font(NDType.body).padding(.horizontal, 12).frame(height: 36)
    }
    private func caption(_ text: String) -> some View {
        Text(text).font(NDType.caption).foregroundStyle(ND.t2).padding(12)
    }
    private func machineList(_ machines: [BrainBarDriveRetention.Machine]) -> some View {
        VStack(spacing: 0) {
            ForEach(Array(machines.enumerated()), id: \.offset) { index, machine in
                if index > 0 { ND.hair.frame(height: 0.5) }
                machineBlock(machine)
            }
        }
    }
    private func machineBlock(_ machine: BrainBarDriveRetention.Machine) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(alignment: .firstTextBaseline) {
                Text(machine.machineId).font(NDType.bodyStrong)
                Spacer()
                if machine.measured { Text(BrainBarDriveRetentionPresentation.summary(machine.onDrive, relativeTo: referenceDate)).font(NDType.caption) }
            }
            if machine.measured {
                // Everything kept first (Keeps, which monthly dates, the unverified and pinned copies held for now), then the
                // decision line last. Unverified copies are drawn only beside kept_by_rule: without it, Keeps falls
                // back to `kept`, which already counts them.
                Text(BrainBarDriveRetentionPresentation.keeps(machine, relativeTo: referenceDate)).foregroundStyle(ND.t2)
                if let representatives = machine.monthlyRepresentatives, !representatives.isEmpty {
                    Text(Say.monthlyDates(representatives.map { BrainBarDriveRetentionPresentation.date($0.date, relativeTo: referenceDate) }.joined(separator: ", ")))
                        .foregroundStyle(ND.t3)
                }
                if machine.keptByRule != nil, let unverified = machine.unverified, unverified.isValid, unverified.count > 0 {
                    Text(Say.unverified(unverified.count, BrainBarDriveRetentionPresentation.size(unverified.bytes)))
                        .foregroundStyle(ND.t2)
                }
                if machine.keptByRule != nil, let pinned = machine.pinned, pinned.isValid, pinned.count > 0 {
                    Text(Say.pinned(pinned.count, BrainBarDriveRetentionPresentation.size(pinned.bytes)))
                        .foregroundStyle(ND.t2)
                }
                Text(Say.wouldRemove(BrainBarDriveRetentionPresentation.summary(machine.wouldRemove, includingRange: true, relativeTo: referenceDate)))
            } else {
                Text(BrainBarDriveRetentionPresentation.status(machine, relativeTo: referenceDate)).foregroundStyle(ND.t2)
            }
        }
        .font(NDType.caption).fixedSize(horizontal: false, vertical: true)
        .frame(maxWidth: .infinity, alignment: .leading).padding(12)
    }
}

/// The confirmation content is built for the future capability; there is intentionally no enable action today.
struct BrainBarDriveRetentionConfirmation: View {
    let presentation: BrainBarDriveRetentionPresentation
    var cancel: () -> Void
    private typealias Say = BrainBarMenuPresenter.DriveRetentionSay
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(presentation.total.map { Say.confirm($0.count, BrainBarDriveRetentionPresentation.size($0.bytes)) } ?? Say.notMeasured)
            Text(Say.trash).font(NDType.caption).foregroundStyle(ND.t2)
            HStack {
                Spacer()
                NDButton(label: Say.cancel, small: true, action: cancel)
                NDButton(label: Say.turnOn, small: true, action: {}).disabled(true)
            }
        }
    }
}
#endif
