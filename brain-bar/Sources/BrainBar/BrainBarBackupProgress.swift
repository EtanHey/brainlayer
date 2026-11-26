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
    /// When the upload process started: an upload can run for many hours (a Mac that slept mid-upload).
    var startedAt: Date? = nil
    var isStalled = false
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
                    checkpointAt: !verifying ? local.bytesConfirmedAt : nil, startedAt: started,
                    isStalled: !verifying && local.bytesConfirmedAt.map { now.timeIntervalSince($0) > checkpointMaxAgeS } == true)
    }

    var fraction: Double? {
        guard let confirmed, let total, total > 0, confirmed >= 0, confirmed <= total else { return nil }
        return Double(confirmed) / Double(total)
    }
    var percentage: Int? { fraction.map { Int(($0 * 100).rounded(.down)) } }
    var accessibilityValue: String {
        if isStalled, let checkpointAt { return checkpointAt.formatted(date: .abbreviated, time: .shortened) }
        return percentage.map { "\($0) percent" } ?? BrainBarMenuPresenter.SurfaceSay.backupInProgress
    }
    var title: String {
        if isStalled { return BrainBarMenuPresenter.SurfaceSay.notUploadingSince }
        return phase == .upload ? BrainBarMenuPresenter.SurfaceSay.backupUploading : BrainBarMenuPresenter.SurfaceSay.backupVerifying
    }
    /// Google Drive's status line in the Pipeline: what the upload is doing, in the station's own words. A fresh
    /// checkpoint is under 20 minutes old by construction, so its time is left to the panel; once it is older the line
    /// itself says since when nothing moved.
    func stationLine(in timeZone: TimeZone = .autoupdatingCurrent) -> String {
        typealias Say = BrainBarMenuPresenter.SurfaceSay
        if isStalled, let checkpointAt { return Say.driveStalledSince(Self.clock(checkpointAt, in: timeZone)) }
        if phase == .verification { return Say.backupVerifyingShort }
        return percentage.map(Say.driveUploadingPercent) ?? Say.backupUploadingShort
    }

    /// "06:26": a clock time as the Pipeline's other lines write it.
    static func clock(_ date: Date, in timeZone: TimeZone) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = timeZone
        f.dateFormat = "HH:mm"
        return f.string(from: date)
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
/// `compact` is the Pipeline's form: the bar alone, exactly its frame's height; its words are Google Drive's status
/// line (`stationLine`) and the panel's rows, so nothing hangs below the station.
struct BrainBarBackupProgressView: View {
    typealias Say = BrainBarMenuPresenter.SurfaceSay
    let progress: BrainBarBackupProgress
    var compact = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var shown: BrainBarBackupProgress?

    var lastGoodText: String? {
        guard !compact, let at = progress.lastGoodAt else { return nil }
        return Say.backupLastGood + " " + at.formatted(date: .abbreviated, time: .shortened)
    }

    var startedText: String? {
        guard !compact, let at = progress.startedAt else { return nil }
        return Say.backupUploadStarted + " " + at.formatted(date: .abbreviated, time: .shortened)
    }

    /// The right-hand words beside the title: the stall's time, or the percentage and when bytes last advanced.
    var checkpointText: String? {
        guard !compact, let at = progress.checkpointAt else { return nil }
        let time = at.formatted(date: .omitted, time: .shortened)
        if progress.isStalled { return time }
        return progress.percentage.map { "\($0)% · " + Say.backupUpdatedAt(time) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            if !compact {
                HStack(spacing: 6) {
                    Text(progress.title).lineLimit(1)
                    Spacer(minLength: 0)
                    if let checkpointText { Text(checkpointText).monospacedDigit().lineLimit(1) }
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
            ForEach([startedText, lastGoodText].compactMap { $0 }, id: \.self) { line in
                Text(line)
                    .foregroundStyle(.secondary)
                    .lineLimit(1).minimumScaleFactor(0.8)
            }
        }
        .font(.system(size: 11))
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
