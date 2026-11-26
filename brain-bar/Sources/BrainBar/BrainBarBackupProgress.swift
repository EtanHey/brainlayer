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
        let stalled = !verifying && local.bytesConfirmedAt.map { now.timeIntervalSince($0) > checkpointMaxAgeS } == true
        // A stalled upload keeps the last confirmed amount (it reached N % at the time its line names); an undated or
        // future-dated checkpoint carries no amount.
        return Self(phase: verifying ? .verification : .upload,
                    identity: "\(local.destination ?? ""):\(started.timeIntervalSince1970)",
                    confirmed: verifying || !(fresh || stalled) ? nil : local.bytesConfirmed, total: verifying ? nil : local.bytesTotal,
                    lastGoodAt: backups.dbSnapshot.flatMap { $0.verified && $0.lastAt <= now ? $0.lastAt : nil },
                    checkpointAt: !verifying ? local.bytesConfirmedAt : nil, startedAt: started,
                    isStalled: stalled)
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
    /// The Backups page's title; a stall says when bytes last moved, in the same words as Drive's line.
    var title: String {
        typealias Say = BrainBarMenuPresenter.SurfaceSay
        if isStalled { return checkpointAt.map { Say.driveStalledAt(Self.clock($0)) } ?? Say.notUploadingSince }
        return phase == .upload ? Say.backupUploading : Say.backupVerifying
    }
    /// Google Drive's status line in the Pipeline: what the upload is doing, in the station's own words. A fresh
    /// checkpoint is under 20 minutes old by construction, so its time is left to the Backups page; once it is older the
    /// line itself says when nothing moved after.
    func stationLine(in timeZone: TimeZone = .autoupdatingCurrent, locale: Locale = .autoupdatingCurrent) -> String {
        typealias Say = BrainBarMenuPresenter.SurfaceSay
        if isStalled, let checkpointAt { return Say.driveStalledAt(Self.clock(checkpointAt, in: timeZone, locale: locale)) }
        if phase == .verification { return Say.backupVerifyingShort }
        return percentage.map(Say.driveUploadingPercent) ?? Say.backupUploadingShort
    }

    /// The one clock style for both surfaces: the person's own 12- or 24-hour setting ("06:26", "6:26 AM").
    static func clock(_ date: Date, in timeZone: TimeZone = .autoupdatingCurrent, locale: Locale = .autoupdatingCurrent) -> String {
        date.formatted(Date.FormatStyle(date: .omitted, time: .shortened, locale: locale, timeZone: timeZone))
    }

    /// What the bar draws. Each state is still and distinct; none is dashed. A moving upload fills in the accent
    /// colour; a stall (or a Drive in error) keeps its amount in grey; a check fills the whole track in a lighter grey;
    /// an unknown amount is the track alone.
    enum BarFill: Equatable, Sendable { case accent(Double), grey(Double), checking, unknown }
    func barFill(muted: Bool = false) -> BarFill {
        if phase == .verification { return .checking }
        guard let fraction else { return .unknown }
        return isStalled || muted ? .grey(fraction) : .accent(fraction)
    }

    func animates(after previous: Self?, reduceMotion: Bool) -> Bool {
        guard !reduceMotion, let previous, phase == .upload, previous.phase == phase, !isStalled, !previous.isStalled,
              identity == previous.identity, total == previous.total,
              let fraction, let old = previous.fraction else { return false }
        return fraction > old
    }
}

/// No timer or looping spinner: motion occurs only when a measured byte checkpoint advances; every other state
/// (stalled, checking, unknown amount) is a still bar, including under Reduce Motion.
/// `compact` is the Pipeline's form: the bar alone, exactly its frame's height; its words are Google Drive's status
/// line (`stationLine`) and the panel's rows, so nothing hangs below the station.
struct BrainBarBackupProgressView: View {
    typealias Say = BrainBarMenuPresenter.SurfaceSay
    let progress: BrainBarBackupProgress
    var compact = false
    /// Google Drive is in error: the amount is drawn in grey, never the accent of a healthy upload.
    var muted = false
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

    /// The right-hand words beside the title: the percentage and when bytes last advanced. A stall says its time in
    /// its title ("Stalled at 7:39"), so nothing is pushed to the right.
    var checkpointText: String? {
        guard !compact, !progress.isStalled, let at = progress.checkpointAt else { return nil }
        return progress.percentage.map { "\($0)% · " + Say.backupUpdatedAt(BrainBarBackupProgress.clock(at)) }
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
                let fill = (shown ?? progress).barFill(muted: muted)
                ZStack(alignment: .leading) {
                    // The unknown amount's track is a little stronger, so it never reads as 0 %.
                    Capsule().fill(Color.secondary.opacity(fill == .unknown ? 0.35 : 0.2))
                    switch fill {
                    case let .accent(fraction): Capsule().fill(Color.accentColor).frame(width: geometry.size.width * fraction)
                    case let .grey(fraction): Capsule().fill(Color.secondary).frame(width: geometry.size.width * fraction)
                    case .checking: Capsule().fill(Color.secondary.opacity(0.6))
                    case .unknown: EmptyView()
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
