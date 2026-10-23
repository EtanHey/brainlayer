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
    var scheduleUnknown = false
    var statusNotice: String? = nil

    var verdictTone: ObservabilityStatusTone {
        if alertTone == .red || checks.contains(where: { $0.tone == .red }) { return .red }
        return statusNotice != nil || scheduleUnknown || checks.contains(where: { $0.tone == .neutral }) ? .neutral : .green
    }

    static func derive(
        _ backups: ObservabilityDocument.Backups,
        now: Date,
        locale: Locale = .current,
        jobStatus: ObservabilityStatusLine? = nil,
        calendar: Calendar = .current,
        statusNotice: String? = nil
    ) -> Self {
        let lines = ObservabilityPresentation.backupStatus(for: backups, locale: locale, jobStatus: jobStatus, now: now, calendar: calendar)
        let rel = { (date: Date) in relative(date, now: now, locale: locale) }

        var upload: String = switch backups.lastVerifiedUpload {
        case let value? where value.verified: "Verified \(rel(value.at))"
        case let value?: "Uploaded \(rel(value.at)), not verified"
        case nil: "No verified upload yet"
        }
        var snapshot: String = switch backups.dbSnapshot {
        case let value? where value.verified: "Verified \(rel(value.lastAt))"
        case let value?: "Saved \(rel(value.lastAt)), not verified"
        case nil: "No verified copy yet"
        }
        if lines.snapshot.tone == .red, backups.dbSnapshot?.verified == true { snapshot = lines.snapshot.text }
        let idle = backups.verifiedIdle(now: now)
        if idle { upload = lines.upload.text }
        let job: String = if let jobStatus { jobStatus.text }
            else if backups.launchd?.bootstrapped == true { "Scheduled" }
            else if backups.launchd?.disabledDirPresent == true { "Paused by a safety stop" }
            else { "Not scheduled" }
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

        var checks = [
            Check(id: "upload", title: "Transcripts in Google Drive", value: upload, tone: lines.upload.tone),
            Check(id: "snapshot", title: "Database copy", value: snapshot, tone: lines.snapshot.tone),
            Check(id: "job", title: "Daily transcript backup", value: job, tone: lines.job.tone),
            Check(id: "freshness", title: lines.freshnessCheckTitle, value: lines.freshnessCheckValue, tone: lines.freshness.tone),
            Check(id: "retention", title: "Safe cleanup", value: retention, tone: lines.retention.tone),
            Check(id: "archives", title: "Verified archives, last 30 days", value: archives, tone: lines.archives.tone),
        ]
        if let annotation = lines.snapshotUpload {
            checks.insert(.init(id: "snapshot-upload", title: "Local database snapshot", value: annotation.text, tone: annotation.tone), at: 2)
        }
        let failing = checks.filter { $0.tone == .red }.count
        let summary: String = if failing == 0 && lines.error?.tone == .red { "Backup alert needs attention" }
            else if failing == 0 && statusNotice != nil { "Backup status needs a fresh check" }
            else if failing == 0 && lines.uploadInProgress { lines.uploadQueued ? "Upload queued" : "Upload in progress" }
            else if failing == 0 && idle { "Idle — no new transcripts" }
            else if jobStatus?.tone == .neutral { jobStatus?.text.hasPrefix("Skipped") == true ? "Skipped"
            : jobStatus?.text.hasPrefix("Deferred") == true ? "Deferred" : "Job schedule unknown" } else { switch failing {
        case 0: "All \(checks.count) checks pass"
        case 1: "1 of \(checks.count) checks needs attention"
        default: "\(failing) of \(checks.count) checks need attention"
        } }

        let alert = alertText(backups.errorType, statusLine: lines.error)
        let alertTone = lines.error?.tone
        let firstRed = checks.first { $0.tone == .red }.map { "\($0.title): \($0.value)" }
        let attention: String? = if lines.errorIsJobAlert, alertTone == .red { alert }
            else { firstRed ?? (alertTone == .red ? alert : nil) }

        return .init(
            summary: summary, checks: checks, alert: alert, alertTone: alertTone,
            details: details(backups, lines: lines, locale: locale, calendar: calendar, jobStatus: jobStatus),
            attentionSentence: attention, scheduleUnknown: jobStatus?.tone == .neutral, statusNotice: statusNotice
        )
    }

    /// "1 hour ago", "3 days ago"; anything under a minute, or in the future (a clock skew), is
    /// "just now".
    static func relative(_ date: Date, now: Date, locale: Locale = .current) -> String {
        ObservabilityPresentation.relative(date, now: now, locale: locale)
    }

    /// The status line's sentence, except that an unrecognised error code reads as a plain
    /// sentence here; the raw code moves to Technical details.
    private static func alertText(_ errorType: String?, statusLine: ObservabilityStatusLine?) -> String? {
        guard let errorType, let statusLine else { return nil }
        if errorType.hasPrefix("jsonl_backup_attempt_"), errorType != "jsonl_backup_attempt_failed" {
            return statusLine.text
        }
        let known = ["drive_credentials_restored_backup_pending", "drive_credentials_missing", "FileNotFoundError"]
        guard !errorType.hasPrefix("job_alert:"), !known.contains(errorType),
              ObservabilityPresentation.dbSnapshotSentence(errorType, thresholdHours: nil) == nil
        else { return statusLine.text }
        return errorType.hasPrefix("jsonl_backup_attempt_")
            ? "The last transcript backup reported an error."
            : "The last database backup reported an error."
    }

    private static func details(
        _ backups: ObservabilityDocument.Backups,
        lines: ObservabilityBackupStatus,
        locale: Locale,
        calendar: Calendar,
        jobStatus: ObservabilityStatusLine?
    ) -> [Detail] {
        func exact(_ date: Date) -> String {
            let formatter = DateFormatter()
            formatter.locale = locale
            formatter.calendar = calendar
            formatter.timeZone = calendar.timeZone
            formatter.dateStyle = .medium
            formatter.timeStyle = .short
            return formatter.string(from: date)
        }
        var rows: [Detail] = []
        if let upload = backups.lastVerifiedUpload {
            rows.append(.init(label: "Drive file ID — latest transcript archive", value: upload.archiveId))
            rows.append(.init(label: "Transcript upload receipt — attempt time", value: exact(upload.at)))
        }
        if let snapshot = backups.dbSnapshot {
            rows.append(.init(label: "Database copy file", value: snapshot.destination))
            rows.append(.init(label: "Offsite database snapshot — created", value: exact(snapshot.lastAt)))
        }
        if let local = backups.localSnapshot {
            if let destination = local.destination { rows.append(.init(label: "Local database snapshot", value: destination)) }
            if let at = local.lastAt { rows.append(.init(label: "Local snapshot created", value: exact(at))) }
        }
        let label = backups.launchd?.label ?? "com.brainlayer.jsonl-backup"
        let jobState = if backups.launchd?.bootstrapped == true { "loaded" }
            else if backups.launchd?.disabledDirPresent == true { "not loaded · parked in .disabled-retention-P0" }
            else { "not loaded" }
        rows.append(.init(label: jobStatus == nil ? "launchd job" : "Job owner", value: jobStatus?.text ?? "\(label) · \(jobState)"))
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
    var copiedDetailID: String? = nil
    let copy: (BrainBarBackupChecks.Detail) -> Void

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
        let verdictTone = checks.verdictTone
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
            if let notice = checks.statusNotice {
                Label(notice, systemImage: "info.circle")
                    .font(.system(size: 11)).foregroundStyle(Color.brainBarTextMuted)
            }
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
                        let isCopied = copiedDetailID == detail.id
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
                            Button { copy(detail) } label: {
                                if isCopied {
                                    Label("Copied", systemImage: "checkmark")
                                } else {
                                    Image(systemName: "doc.on.doc")
                                }
                            }
                                .buttonStyle(.borderless)
                                .controlSize(.small)
                                .help(isCopied ? "Copied" : "Copy")
                                .accessibilityLabel("\(isCopied ? "Copied" : "Copy") \(detail.label)")
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
