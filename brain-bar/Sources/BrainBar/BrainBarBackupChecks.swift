import Foundation
import SwiftUI

/// Etan row E1 (2026-10-01): the Backups page's checks in plain words. Each check carries the tone
/// of the status line it replaces (#1029 B1), so the badge and the list can never disagree. Drive
/// IDs, file names and the launchd label live only in `details`, behind a disclosure with Copy.
struct BrainBarBackupChecks: Equatable, Sendable {
    struct Check: Equatable, Identifiable, Sendable {
        let id: String
        let title: String
        let value: String
        let tone: ObservabilityStatusTone
    }

    struct Detail: Equatable, Identifiable, Sendable {
        var id: String { label }
        let label: String
        let value: String
    }

    let summary: String
    let checks: [Check]
    /// A job alert or backup error, in its own words. Nil when there is none.
    let alert: String?
    let alertTone: ObservabilityStatusTone?
    let details: [Detail]
    /// The one sentence that speaks for the backups, chosen exactly as
    /// `ObservabilityBackupStatus.attentionLine` chooses its line: a job alert first, then the first
    /// red check, then a red alert.
    let attentionSentence: String?

    static func derive(
        _ backups: ObservabilityDocument.Backups,
        now: Date,
        locale: Locale = .current
    ) -> Self {
        let lines = ObservabilityPresentation.backupStatus(for: backups, locale: locale)
        let rel = { (date: Date) in relative(date, now: now, locale: locale) }

        let upload: String = switch backups.lastVerifiedUpload {
        case let value? where value.verified: "Verified \(rel(value.at))"
        case let value?: "Uploaded \(rel(value.at)), not verified"
        case nil: "No verified upload yet"
        }
        let snapshot: String = switch backups.dbSnapshot {
        case let value? where value.verified: "Verified \(rel(value.lastAt))"
        case let value?: "Saved \(rel(value.lastAt)), not verified"
        case nil: "No verified copy yet"
        }
        let job: String = if backups.launchd?.bootstrapped == true { "Scheduled" }
            else if backups.launchd?.disabledDirPresent == true { "Paused by a safety stop" }
            else { "Not scheduled" }
        let threshold = backups.thresholdHours.map { "\(Int($0)) hours" }
        let freshness: String = switch backups.freshness {
        case "fresh": threshold.map { "Both copies are under \($0) old" } ?? "Both copies are recent"
        case "stale": threshold.map { "A copy is older than \($0)" } ?? "A copy is out of date"
        default: "Unknown"
        }
        let retention: String = switch backups.retentionInvariant {
        case "PASS": "On"
        case "FAIL": "Failing"
        default: "Unknown"
        }
        let archives: String = switch backups.survivingArchives30D {
        case nil: "Unknown"
        case 0?: "None"
        case let count?: ObservabilityPresentation.number(count, locale: locale)
        }

        let checks = [
            Check(id: "upload", title: "Transcripts in Google Drive", value: upload, tone: lines.upload.tone),
            Check(id: "snapshot", title: "Database copy", value: snapshot, tone: lines.snapshot.tone),
            Check(id: "job", title: "Daily transcript backup", value: job, tone: lines.job.tone),
            Check(id: "freshness", title: "Up to date", value: freshness, tone: lines.freshness.tone),
            Check(id: "retention", title: "Safe cleanup", value: retention, tone: lines.retention.tone),
            Check(id: "archives", title: "Verified archives, last 30 days", value: archives, tone: lines.archives.tone),
        ]
        let failing = checks.filter { $0.tone == .red }.count
        let summary = switch failing {
        case 0: "All \(checks.count) checks pass"
        case 1: "1 of \(checks.count) checks needs attention"
        default: "\(failing) of \(checks.count) checks need attention"
        }

        let alert = alertText(backups.errorType, statusLine: lines.error)
        let alertTone = lines.error?.tone
        let firstRed = checks.first { $0.tone == .red }.map { "\($0.title): \($0.value)" }
        let attention: String? = if lines.errorIsJobAlert, alertTone == .red { alert }
            else { firstRed ?? (alertTone == .red ? alert : nil) }

        return .init(
            summary: summary, checks: checks, alert: alert, alertTone: alertTone,
            details: details(backups, lines: lines, locale: locale),
            attentionSentence: attention
        )
    }

    /// "1 hour ago", "3 days ago"; anything under a minute, or in the future (a clock skew), is
    /// "just now".
    static func relative(_ date: Date, now: Date, locale: Locale = .current) -> String {
        guard now.timeIntervalSince(date) >= 60 else { return "just now" }
        let formatter = RelativeDateTimeFormatter()
        formatter.locale = locale
        formatter.unitsStyle = .full
        return formatter.localizedString(for: date, relativeTo: now)
    }

    /// The status line's sentence, except that an unrecognised error code reads as a plain
    /// sentence here; the raw code moves to Technical details.
    private static func alertText(_ errorType: String?, statusLine: ObservabilityStatusLine?) -> String? {
        guard let errorType, let statusLine else { return nil }
        let known = ["drive_credentials_restored_backup_pending", "drive_credentials_missing", "FileNotFoundError"]
        guard !errorType.hasPrefix("job_alert:"), !known.contains(errorType) else { return statusLine.text }
        return errorType.hasPrefix("jsonl_backup_attempt_")
            ? "The last transcript backup reported an error."
            : "The last database backup reported an error."
    }

    private static func details(
        _ backups: ObservabilityDocument.Backups,
        lines: ObservabilityBackupStatus,
        locale: Locale
    ) -> [Detail] {
        func exact(_ date: Date) -> String {
            date.formatted(Date.FormatStyle(date: .abbreviated, time: .shortened).locale(locale))
        }
        var rows: [Detail] = []
        if let upload = backups.lastVerifiedUpload {
            rows.append(.init(label: "Transcript archive ID", value: upload.archiveId))
            rows.append(.init(label: "Last transcript upload", value: exact(upload.at)))
        }
        if let snapshot = backups.dbSnapshot {
            rows.append(.init(label: "Database copy file", value: snapshot.destination))
            rows.append(.init(label: "Last database copy", value: exact(snapshot.lastAt)))
        }
        let label = backups.launchd?.label ?? "com.brainlayer.jsonl-backup"
        let jobState = if backups.launchd?.bootstrapped == true { "loaded" }
            else if backups.launchd?.disabledDirPresent == true { "not loaded · parked in .disabled-retention-P0" }
            else { "not loaded" }
        rows.append(.init(label: "launchd job", value: "\(label) · \(jobState)"))
        let threshold = backups.thresholdHours.map { " · threshold \(Int($0)) h" } ?? ""
        rows.append(.init(label: "Freshness", value: "\(backups.freshness ?? "unknown")\(threshold)"))
        rows.append(.init(label: "Retention invariant", value: backups.retentionInvariant ?? "unknown"))
        if let errorType = backups.errorType, !errorType.isEmpty, !lines.errorIsJobAlert {
            rows.append(.init(label: "Error code", value: errorType))
        }
        return rows
    }
}

/// E1: the Backups page's recovery checks, in the page's card language: a title with a verdict
/// badge, a one-line summary, the checks as label/value rows, and the raw identifiers behind a
/// Technical details disclosure where every value is selectable and has Copy.
struct BrainBarBackupChecksCard: View {
    let checks: BrainBarBackupChecks
    /// An alert the page already shows elsewhere (the job-alert card); never repeated here.
    let hiddenAlert: String?
    @Binding var detailsExpanded: Bool
    let copy: (String) -> Void

    private static func color(_ tone: ObservabilityStatusTone) -> Color {
        switch tone {
        case .green: Color(nsColor: BrainBarDesignTokens.Colors.statusOK)
        case .red: Color(nsColor: BrainBarDesignTokens.Colors.statusError)
        case .neutral: Color(nsColor: BrainBarDesignTokens.Colors.statusUnknown)
        }
    }

    private static func symbol(_ tone: ObservabilityStatusTone) -> String {
        switch tone {
        case .green: "checkmark.circle.fill"
        case .red: "exclamationmark.triangle.fill"
        case .neutral: "info.circle.fill"
        }
    }

    var body: some View {
        let failing = checks.checks.contains { $0.tone == .red }
        let verdictTone: ObservabilityStatusTone = failing ? .red : .green
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text("Recovery checks").font(.system(size: 13, weight: .semibold))
                Spacer(minLength: 12)
                Label(checks.summary, systemImage: Self.symbol(verdictTone))
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(Self.color(verdictTone))
            }
            Text("Whether each copy exists, is recent, and was verified.")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color.brainBarTextMuted)
            if let alert = checks.alert, alert != hiddenAlert, let tone = checks.alertTone {
                Label(alert, systemImage: Self.symbol(tone))
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Self.color(tone))
                    .fixedSize(horizontal: false, vertical: true)
            }
            VStack(spacing: 0) {
                ForEach(Array(checks.checks.enumerated()), id: \.element.id) { index, check in
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Image(systemName: Self.symbol(check.tone))
                            .font(.system(size: 11))
                            .foregroundStyle(Self.color(check.tone))
                        Text(check.title)
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(Color.brainBarTextPrimary)
                        Spacer(minLength: 12)
                        Text(check.value)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(check.tone == .red ? Self.color(.red) : Color.brainBarTextSecondary)
                            .multilineTextAlignment(.trailing)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .accessibilityElement(children: .combine)
                    .accessibilityIdentifier("brainbar.backups.check.\(check.id)")
                    if index < checks.checks.count - 1 {
                        Rectangle().fill(Color.brainBarBorderSoft).frame(height: 1)
                    }
                }
            }
            .background(RoundedRectangle(cornerRadius: BrainBarDesignTokens.Radius.sm, style: .continuous).fill(Color.brainBarGlassSecondary))
            .overlay(RoundedRectangle(cornerRadius: BrainBarDesignTokens.Radius.sm, style: .continuous).strokeBorder(Color.brainBarBorderSoft))
            Button {
                detailsExpanded.toggle()
            } label: {
                Label("Technical details", systemImage: detailsExpanded ? "chevron.down" : "chevron.right")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Color.brainBarTextMuted)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("brainbar.backups.technical-details")
            if detailsExpanded {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(checks.details) { detail in
                        HStack(alignment: .firstTextBaseline, spacing: 12) {
                            Text(detail.label.uppercased())
                                .font(.system(size: 9, weight: .bold))
                                .tracking(0.6)
                                .foregroundStyle(Color.brainBarTextMuted)
                                .frame(width: 150, alignment: .leading)
                            Text(detail.value)
                                .font(.system(size: 10, weight: .medium, design: .monospaced))
                                .foregroundStyle(Color.brainBarTextSecondary)
                                .textSelection(.enabled)
                                .fixedSize(horizontal: false, vertical: true)
                            Spacer(minLength: 8)
                            Button { copy(detail.value) } label: { Image(systemName: "doc.on.doc") }
                                .buttonStyle(.borderless)
                                .controlSize(.small)
                                .help("Copy")
                                .accessibilityLabel("Copy \(detail.label)")
                        }
                    }
                }
                .padding(.leading, 18)
            }
        }
        .padding(.vertical, 6)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("brainbar.backups.recovery-checks")
    }
}
