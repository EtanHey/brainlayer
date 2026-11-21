import SwiftUI

/// Checkpoint facts only. A live process is required: an interrupted checkpoint is not running work.
struct BrainBarBackupProgress: Equatable, Sendable {
    enum Phase: Equatable, Sendable { case upload, verification }
    var phase: Phase
    var identity: String
    var confirmed: Int64?
    var total: Int64?
    var lastGoodAt: Date?
    var checkpointAt: Date? = nil
    static let checkpointMaxAgeS: TimeInterval = 20 * 60

    static func read(_ backups: ObservabilityDocument.Backups, now: Date) -> Self? {
        guard backups.state == "measured", let local = backups.localSnapshot,
              local.uploadProcessLive == true,
              local.uploadError?.isEmpty != false,
              let started = local.uploadProcessStartedAt, started <= now else { return nil }
        let verifying = local.uploaded && local.verificationPending == true
        guard local.uploadState == "in_progress" || (local.uploadState == "pending" && verifying) else { return nil }
        guard !local.uploaded || verifying else { return nil }
        let fresh = local.bytesConfirmedAt.map { (0...checkpointMaxAgeS).contains(now.timeIntervalSince($0)) } == true
        return Self(phase: verifying ? .verification : .upload,
                    identity: "\(local.destination ?? ""):\(started.timeIntervalSince1970)",
                    confirmed: verifying || !fresh ? nil : local.bytesConfirmed, total: verifying ? nil : local.bytesTotal,
                    lastGoodAt: backups.dbSnapshot.flatMap { $0.verified && $0.lastAt <= now ? $0.lastAt : nil },
                    checkpointAt: fresh && !verifying ? local.bytesConfirmedAt : nil)
    }

    var fraction: Double? {
        guard let confirmed, let total, total > 0, confirmed >= 0, confirmed <= total else { return nil }
        return Double(confirmed) / Double(total)
    }
    var percentage: Int? { fraction.map { Int(($0 * 100).rounded(.down)) } }
    var accessibilityValue: String { percentage.map { "\($0) percent" } ?? BrainBarMenuPresenter.SurfaceSay.backupInProgress }
    var title: String {
        phase == .upload ? BrainBarMenuPresenter.SurfaceSay.backupUploading : BrainBarMenuPresenter.SurfaceSay.backupVerifying
    }
    func animates(after previous: Self?, reduceMotion: Bool) -> Bool {
        guard !reduceMotion, let previous, phase == .upload, previous.phase == phase,
              identity == previous.identity, total == previous.total,
              let fraction, let old = previous.fraction else { return false }
        return fraction > old
    }
}

/// No timer or looping spinner: motion occurs only when a measured byte checkpoint advances.
/// The open-ended bar stays still when a total is unavailable, including under Reduce Motion.
struct BrainBarBackupProgressView: View {
    let progress: BrainBarBackupProgress
    var compact = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var shown: BrainBarBackupProgress?

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                Text(compact ? (progress.phase == .upload ? BrainBarMenuPresenter.SurfaceSay.backupUploadingShort
                    : BrainBarMenuPresenter.SurfaceSay.backupVerifyingShort) : progress.title).lineLimit(1)
                Spacer(minLength: 0)
                if let percent = progress.percentage, let at = progress.checkpointAt {
                    Text("\(percent)% · " + at.formatted(date: .omitted, time: .shortened)).monospacedDigit().lineLimit(1)
                }
            }
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.secondary.opacity(0.2))
                    if let fraction = (shown ?? progress).fraction {
                        Capsule().fill(Color.accentColor).frame(width: geometry.size.width * fraction)
                    } else {
                        Capsule().strokeBorder(Color.secondary, style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
                    }
                }
            }
            .frame(height: 3)
            if let at = progress.lastGoodAt {
                Text((compact ? BrainBarMenuPresenter.SurfaceSay.backupLastGoodShort : BrainBarMenuPresenter.SurfaceSay.backupLastGood)
                    + " " + at.formatted(date: .abbreviated, time: .shortened))
                    .foregroundStyle(.secondary)
                    .lineLimit(1).minimumScaleFactor(0.8)
            }
        }
        .font(.system(size: compact ? 10 : 11))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(progress.title)
        .accessibilityValue(progress.accessibilityValue)
        .onAppear { shown = progress }
        .onChange(of: progress) { old, next in
            withAnimation(next.animates(after: old, reduceMotion: reduceMotion) ? .easeOut(duration: 0.25) : nil) {
                shown = next
            }
        }
    }
}
