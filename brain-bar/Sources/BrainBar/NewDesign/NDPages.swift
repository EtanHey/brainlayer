import SwiftUI

private typealias Say = BrainBarMenuPresenter.SurfaceSay

/// The rails' axis (§ 2.4, § 2.5): tick labels centred on their time inside the rail, then the column heads.
struct NDRailAxis: View {
    let ticks: [(label: String, hours: Double)]
    let past: Double
    let ahead: Double
    let leading: CGFloat
    let heads: [(String, CGFloat)]

    var body: some View {
        HStack(spacing: 0) {
            Spacer().frame(width: leading)
            GeometryReader { geo in
                let width = geo.size.width
                ForEach(Array(ticks.enumerated()), id: \.offset) { _, tick in
                    let x = CGFloat((tick.hours + past) / (past + ahead)) * width
                    Text(tick.label).font(NDType.captionNumber).foregroundStyle(ND.t3).fixedSize()
                        .alignmentGuide(.leading) { d in
                            let left = min(max(x - d.width / 2, 0), width - d.width)
                            return -left
                        }
                }
                .frame(width: width, height: 18, alignment: .bottomLeading)
            }
            ForEach(Array(heads.enumerated()), id: \.offset) { _, head in
                Text(head.0).font(NDType.caption).foregroundStyle(ND.t3).frame(width: head.1, alignment: .trailing)
            }
        }
        .frame(height: 18, alignment: .bottom)
        .accessibilityHidden(true)
    }
}

/// A legend item: its mark, then its words (caption, t2).
struct NDLegendItem<Mark: View>: View {
    let words: String
    @ViewBuilder let mark: Mark
    var body: some View {
        HStack(spacing: 6) {
            mark
            Text(words).font(NDType.caption).foregroundStyle(ND.t2)
        }
    }
}

/// The page's fault headline under its title (Backups, Jobs): symbol, title and detail; the action right-aligned.
struct NDPageFault: View {
    @ObservedObject var store: NDStore
    let item: NDR1.Item

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: item.severity == .red ? "exclamationmark.circle.fill" : "exclamationmark.triangle.fill")
                .font(.system(size: 15)).foregroundStyle(item.severity == .red ? ND.red : ND.orange).frame(width: 16, height: 22)
            VStack(alignment: .leading, spacing: 2) {
                Text(item.title).font(NDType.title).tracking(-0.25).foregroundStyle(ND.t1).frame(minHeight: 22)
                if let detail = item.detail, !detail.isEmpty {
                    Text(keepingUnits(detail)).font(NDType.body).foregroundStyle(ND.t2)
                }
            }
            Spacer(minLength: 16)
            if let action = item.action {
                NDWaitingSwap(waiting: NDDriveWaiting.shows(store.inputs, for: action)) {
                    NDButton(label: action.label, filled: NDButtonPolicy.filled(action)) { store.perform(action.kind) }
                } line: {
                    Text(Say.waitingForBrowser).font(NDType.body).foregroundStyle(ND.t2)
                }
            }
        }
    }
}

/// One filled button per surface: the action that only the owner can take (Reconnect / Connect). A copied report is
/// always the bordered, secondary button (lead heads-up for revision 1.1, point 5; comp 16).
enum NDButtonPolicy {
    static func filled(_ action: BrainBarMenuPresentation.Action) -> Bool {
        [.reconnectDrive, .connectDrive].contains(action.kind)
    }
}

// MARK: - Backups (§ 2.4)

struct NDBackupsPage: View {
    @ObservedObject var store: NDStore
    static let past: Double = 144
    static let ahead: Double = 24
    /// R1.7 item 4: the "late after" line is the check's own amber threshold (health_verdict.py), the same one that
    /// makes the copy on this Mac amber.
    static let limit: Double = -NDBackupFacts.lateAfterHours
    static let symbolColumn: CGFloat = 24
    static let nameColumn: CGFloat = 196
    static let valueColumn: CGFloat = 84
    static let symbols: [String: String] = ["backup_local": "laptopcomputer", "backup_offsite": "icloud",
                                            "transcripts": "text.bubble", "integrity": "checkmark.shield"]
    static let pageChecks: Set<String> = ["drive_auth", "backup_local", "backup_offsite", "transcripts", "integrity", "disk"]

    var body: some View {
        let m = store.model
        let r1 = store.r1
        let lanes = lanes(m)
        let fault = r1.items.first { !$0.concerns.isDisjoint(with: Self.pageChecks) }
        let legend = Self.legend(lanes)
        let failed = legend.contains(Say.failed)
        let stalled = legend.contains(Say.notUploadingSince)
        VStack(alignment: .leading, spacing: 0) {
            NDPageTop(title: Say.backups, detail: fault == nil ? Say.backupsDetail : nil)
            if let fault { NDPageFault(store: store, item: fault).padding(.top, 14) }
            NDCard {
                VStack(alignment: .leading, spacing: 0) {
                    HStack(spacing: 20) {
                        NDLegendItem(words: Say.copied) { Circle().fill(ND.t1).frame(width: 7, height: 7) }
                        NDLegendItem(words: Say.nextRun) { Circle().stroke(ND.t2, lineWidth: 1.5).frame(width: 6, height: 6) }
                        NDLegendItem(words: Say.lateIfLeftOfLine) { ND.t3.frame(width: 1, height: 12) }
                        if failed { NDLegendItem(words: Say.failed) { Circle().fill(ND.red).frame(width: 7, height: 7) } }
                        if legend.contains(Say.failedThenRetried) {
                            NDLegendItem(words: Say.failedThenRetried) { Circle().stroke(ND.red, lineWidth: 1.4).frame(width: 6.4, height: 6.4) }
                        }
                        if stalled {
                            NDLegendItem(words: Say.notUploadingSince) {
                                Canvas { ctx, size in
                                    var p = Path(); p.move(to: CGPoint(x: 0, y: size.height / 2)); p.addLine(to: CGPoint(x: size.width, y: size.height / 2))
                                    ctx.stroke(p, with: .color(ND.red), style: StrokeStyle(lineWidth: 1.5, dash: [3, 3]))
                                }
                                .frame(width: 14, height: 6)
                            }
                        }
                    }
                    .frame(height: 18)
                    
                    NDRailAxis(ticks: [(Say.daysAgoAxis(6), -144), (Say.daysAgoAxis(3), -72), (Say.lateAfter(NDFormat.ageText(NDBackupFacts.lateAfterHours * 3600)), Self.limit), (Say.now, 0),
                                       (Say.inHoursAxis(24), 24)],
                               past: Self.past, ahead: Self.ahead, leading: Self.symbolColumn + Self.nameColumn,
                               heads: [(Say.newest, Self.valueColumn), (Say.next, Self.valueColumn)])
                        .padding(.top, 12)
                    VStack(spacing: 0) {
                        ForEach(lanes) { NDBackupLaneView(lane: $0) }
                    }
                    .padding(.top, 4)
                    .background(alignment: .topLeading) { limitLine }
                }
            }
            .padding(.top, fault == nil ? 20 : 18)
            details(m).padding(.top, 14).padding(.horizontal, 2)
        }
    }

    /// The legend's words: always copied, next run and the late line; "failed" whenever a red dot is drawn (so a past
    /// failure under "All good" is said in words), and "not uploading since" whenever a red dashed line is.
    static func legend(_ lanes: [Lane]) -> [String] {
        var words = [Say.copied, Say.nextRun, Say.lateIfLeftOfLine]
        if lanes.contains(where: { $0.events.contains { $0.kind == .fail } }) { words.append(Say.failed) }
        if lanes.contains(where: { $0.events.contains { $0.kind == .retried } }) { words.append(Say.failedThenRetried) }
        if lanes.contains(where: { $0.stalledFrom != nil }) { words.append(Say.notUploadingSince) }
        return words
    }

    /// "a copy older than this line is late": one line through every lane, at the amber threshold.
    private var limitLine: some View {
        GeometryReader { geo in
            let railWidth = geo.size.width - Self.symbolColumn - Self.nameColumn - 2 * Self.valueColumn
            let x = Self.symbolColumn + Self.nameColumn + CGFloat((Self.limit + Self.past) / (Self.past + Self.ahead)) * railWidth
            ND.t3.opacity(0.55).frame(width: 1, height: geo.size.height).offset(x: x)
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    /// Google Drive / Newest copy on this Mac / Room for backups, below a hairline.
    @ViewBuilder private func details(_ m: NDModel) -> some View {
        let rows = m.areas.flatMap(\.rows)
        let auth = rows.first { $0.id == "drive_auth" }
        let signedOut = auth?.number.tone == .fault
        let account = signedOut ? Say.signedOutValue : auth?.number.value == "—" || auth == nil ? NDModel.Number.unknown.value : Say.signedIn
        let disk = rows.first { $0.id == "disk" }
        let room: String = switch disk?.number.tone {
        case .fault?: Say.full
        case .warn?: Say.tight
        case .fine?: disk?.number.value == "—" ? NDModel.Number.unknown.value : Say.enough
        case nil: NDModel.Number.unknown.value
        }
        VStack(spacing: 0) {
            detail(Say.stageDrive, account, tone: signedOut ? .fault : .fine)
            detail(Say.newestCopyOnMac, newestCopyFact(m)) {
                if store.inputs.canRevealLocalCopy {
                    NDButton(label: Say.showInFinder, small: true) { store.revealLocalCopy() }
                }
            }
            detail(Say.roomForBackups, room, tone: disk?.number.tone ?? .fine)
        }
    }

    private func detail(_ label: String, _ value: String, tone: NDModel.Tone = .fine) -> some View {
        detail(label, value, tone: tone) { EmptyView() }
    }

    private func detail<A: View>(_ label: String, _ value: String, tone: NDModel.Tone = .fine, @ViewBuilder action: () -> A) -> some View {
        HStack(spacing: 0) {
            Text(label).font(NDType.body).foregroundStyle(ND.t2).frame(width: Self.symbolColumn + Self.nameColumn, alignment: .leading)
            Text(value).font(tone == .fault ? NDType.bodyStrong : NDType.body)
                .foregroundStyle(tone == .fault ? ND.redText : tone == .warn ? ND.orangeText : ND.t1)
            Spacer()
            action()
        }
        .frame(height: NDGrid.row)
    }

    struct Lane: Identifiable {
        var id: String
        var name: String
        var events: [NDRail.Event]
        var unknown: Bool
        var stalledFrom: Double?
        /// Review N1: set when the newest cell is unknown; the rail claims no confirmed copy.
        var unconfirmedFrom: Double? = nil
        /// R1.2 change 13: the retried ring's help tag ("Upload failed at 03:16; the 04:02 retry succeeded.").
        var help: String? = nil
        var number: NDModel.Number
        var strong: Bool
        var accessibility: String
    }

    func lanes(_ m: NDModel) -> [Lane] {
        let now = store.inputs.capture.now
        let rows = m.areas.first { $0.thing == .safe }?.rows ?? []
        let lanes: NDBackupLanes? = if case let .read(l) = store.inputs.backupHistory { l } else { nil }
        let jobs = store.inputs.jobs ?? []
        let databaseNext = jobs.first { $0.job == .backupDaily }?.nextRun ?? store.inputs.capture.backupDailyNextRunAt
        let conversationsNext = jobs.first { $0.job == .jsonlBackup }?.nextRun
        let driveDown = m.faults.contains { $0.concerns.contains("drive_auth") }
        func lane(_ id: String, _ name: String, _ attempts: [NDAttempt]?, next: Date?, stalls: Bool) -> Lane {
            let row = rows.first { $0.id == id }
            // R1.2 change 13: a failed attempt followed by a success within the same night (12 h) is one hollow red
            // ring ("failed, then copied on retry"); a failure with no later success that night stays a filled red dot.
            let sorted = (attempts ?? []).sorted { $0.at < $1.at }
            var events: [NDRail.Event] = []
            var retryHelp: [String] = []
            var skip: Set<Int> = []
            for (index, attempt) in sorted.enumerated() where !skip.contains(index) {
                let hours = attempt.at.timeIntervalSince(now) / 3600
                if !attempt.ok, let retry = sorted.indices.first(where: { $0 > index && sorted[$0].ok
                        && sorted[$0].at.timeIntervalSince(attempt.at) <= 12 * 3600 }) {
                    skip.insert(retry)
                    events.append(.init(hours: hours, kind: .retried))
                    retryHelp.append(Say.uploadFailedRetried(NDR1.hhmm(attempt.at, store.inputs.capture),
                                                             NDR1.hhmm(sorted[retry].at, store.inputs.capture)))
                } else {
                    events.append(.init(hours: hours, kind: attempt.ok ? .ok : .fail))
                }
            }
            events = events.filter { $0.hours >= -Self.past }
            let stalled = stalls && driveDown
            let last = events.map(\.hours).max()
            if let next, !stalled { events.append(.init(hours: next.timeIntervalSince(now) / 3600, kind: .next)) }
            let count = attempts?.count ?? 0, failed = attempts?.filter { !$0.ok }.count ?? 0
            let spoken = attempts == nil ? "\(name): the record could not be read"
                : "\(name): \(count) \(count == 1 ? "attempt" : "attempts") this week, \(failed) failed"
            let newest = row.map { ", newest " + ($0.number.value == "—" ? "not known" : $0.number.text) } ?? ""
            let number = (row?.strong ?? false) ? faultNumber(row?.number) : row?.number ?? .unknown
            // Review N1: when the newest cell is unknown, the attempts keep their places (the log is the app's own read,
            // L2) but none of them is drawn as a confirmed copy, and the line from the first one to now is dashed.
            let unconfirmed = number.value == NDModel.Number.unknown.value
            var unconfirmedFrom: Double? = nil
            if unconfirmed {
                events = events.map { $0.kind == .next ? $0 : NDRail.Event(hours: $0.hours, kind: .unconfirmed) }
                unconfirmedFrom = events.filter { $0.kind != .next }.map(\.hours).min()
            }
            return Lane(id: id, name: name, events: events, unknown: attempts == nil,
                        stalledFrom: stalled ? (last ?? -Self.past) : nil, unconfirmedFrom: unconfirmedFrom,
                        help: retryHelp.isEmpty ? nil : retryHelp.joined(separator: " "),
                        number: number, strong: row?.strong ?? false, accessibility: spoken + newest)
        }
        return [
            lane("backup_local", NDPipelineLayout.name(.mac), lanes?.local, next: databaseNext, stalls: false),  // R1.10.5: one name
            lane("backup_offsite", "Copy in Google Drive", lanes?.drive, next: databaseNext, stalls: true),
            lane("transcripts", "Conversations in Drive", lanes?.conversations, next: conversationsNext, stalls: true),
            lane("integrity", "Database check", lanes?.integrity, next: databaseNext, stalls: false),
        ]
    }

    /// On this page a concerned copy's age is the fault itself (comp 12): Fault Semibold.
    private func faultNumber(_ n: NDModel.Number?) -> NDModel.Number {
        guard var n else { return .unknown }
        n.tone = .fault
        n.strong = true
        return n
    }


    /// B3: "Newest copy" follows the "Copy on this Mac" lane: one reading, one staleness rule. When the lane's age is
    /// unknown (a stale or missing check), the time is unknown too.
    func newestCopyFact(_ m: NDModel) -> String {
        let lane = lanes(m).first { $0.id == "backup_local" }
        guard let at = store.inputs.newestLocalCopyAt, let lane, lane.number.value != NDModel.Number.unknown.value else { return NDModel.Number.unknown.value }
        return NDFormat.day(at, now: store.inputs.capture.now, zone: store.inputs.capture.timeZone).capitalizedFirst
    }

}

/// One copy at 24 pt: `24 symbol | 196 name | rail | 84 Newest | 84 Next`.
struct NDBackupLaneView: View {
    let lane: NDBackupsPage.Lane

    var body: some View {
        let next = lane.events.first { $0.kind == .next }
        let newest: String = {
            if lane.number.value == NDModel.Number.unknown.value || lane.number.unit.isEmpty { return lane.number.value }
            return Say.ago(lane.number.text)
        }()
        HStack(spacing: 0) {
            Image(systemName: NDBackupsPage.symbols[lane.id] ?? "circle").font(.system(size: 13))
                .foregroundStyle(lane.strong ? ND.red : ND.t2).frame(width: NDBackupsPage.symbolColumn, alignment: .leading)
            Text(lane.name).font(lane.strong ? NDType.bodyStrong : NDType.body).foregroundStyle(ND.t1)
                .frame(width: NDBackupsPage.nameColumn, alignment: .leading)
            NDRail(past: NDBackupsPage.past, ahead: NDBackupsPage.ahead, events: lane.events, unknown: lane.unknown,
                   stalledFrom: lane.stalledFrom, unconfirmedFrom: lane.unconfirmedFrom)
            Text(newest).font(lane.strong ? NDType.numberStrong : NDType.number)
                .foregroundStyle(lane.strong ? ND.redText : ND.t2)
                .frame(width: NDBackupsPage.valueColumn, alignment: .trailing)
            Text(next.map { Say.inAge(NDFormat.ageText($0.hours * 3600)) } ?? NDModel.Number.unknown.value).font(NDType.number)
                .foregroundStyle(ND.t2)
                .frame(width: NDBackupsPage.valueColumn, alignment: .trailing)
        }
        .frame(height: NDGrid.row)
        .help(lane.help ?? "")
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(lane.accessibility + (lane.help.map { ". " + $0 } ?? ""))
    }
}

/// A link: Lumen text, dimmed while pressed; it takes keyboard focus with the system ring (fix 12).
struct NDLinkButtonStyle: PrimitiveButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        Button(action: configuration.trigger) { configuration.label }
            .buttonStyle(Pressable())
            .ndKeyboardFocus(action: configuration.trigger)
    }

    private struct Pressable: ButtonStyle {
        func makeBody(configuration: Configuration) -> some View {
            configuration.label.font(.system(size: 13, weight: .medium)).foregroundStyle(ND.lumen)
                .opacity(configuration.isPressed ? 0.7 : 1)
                .contentShape(Rectangle())
        }
    }
}

extension View {
    /// Review fix 9 / brief fix 12: Tab reaches the window's buttons, range control and links with or without Full
    /// Keyboard Access, the system draws its focus ring, and Space or Return does what a click does.
    func ndKeyboardFocus(action: @escaping () -> Void) -> some View {
        focusable(interactions: .activate)
            .onKeyPress(.space) { action(); return .handled }
            .onKeyPress(.return) { action(); return .handled }
    }
}

// MARK: - Jobs (§ 2.5)

struct NDJobsPage: View {
    @ObservedObject var store: NDStore
    static let past: Double = 36
    static let ahead: Double = 12
    static let nameColumn: CGFloat = 220
    static let valueColumn: CGFloat = 84
    static let switchColumn: CGFloat = 44
    /// Jobs that run every few minutes: a dotted band to now, no dots (no log records each run).
    static let frequent: Set<BrainLayerLaunchdJob> = [.drain, .walCheckpoint]

    var body: some View {
        let r1 = store.r1
        let fault = r1.items.first { item in ["nightly", "weekly"].contains { item.id == $0 || item.concerns.contains($0) } }
        VStack(alignment: .leading, spacing: 0) {
            NDPageTop(title: Say.jobs, detail: fault == nil ? Say.jobsDetail : nil)
            if let fault { NDPageFault(store: store, item: fault).padding(.top, 14) }
            NDCard {
                VStack(alignment: .leading, spacing: 0) {
                    HStack(spacing: 20) {
                        NDLegendItem(words: Say.ran) { Circle().fill(ND.t1).frame(width: 7, height: 7) }
                        NDLegendItem(words: Say.failed) { Circle().fill(ND.red).frame(width: 7, height: 7) }
                        NDLegendItem(words: Say.runsAllTheTime) { Capsule().fill(ND.t2).frame(width: 14, height: 3) }
                        NDLegendItem(words: Say.everyFewMinutes) {
                            HStack(spacing: 2) { ForEach(0..<3, id: \.self) { _ in Circle().fill(ND.t2).frame(width: 2.5, height: 2.5) } }
                        }
                        NDLegendItem(words: Say.nextRun) { Circle().stroke(ND.t2, lineWidth: 1.5).frame(width: 6, height: 6) }
                        NDLegendItem(words: Say.outsideTimeline) {
                            Image(systemName: "chevron.left").font(.system(size: 9, weight: .semibold)).foregroundStyle(ND.t2)
                        }
                    }
                    .frame(height: 18)
                    NDRailAxis(ticks: [(Say.hoursAgoAxis(36), -36), (Say.hoursAgoAxis(24), -24), (Say.hoursAgoAxis(12), -12), (Say.now, 0),
                                       (Say.inHoursAxis(12), 12)],
                               past: Self.past, ahead: Self.ahead, leading: Self.nameColumn,
                               heads: [(Say.lastRun, Self.valueColumn), (Say.nextRunHead, Self.valueColumn), (BrainBarMenuPresenter.SurfaceSay.onHeader, Self.switchColumn)])
                        .padding(.top, 12)
                    VStack(spacing: 0) {
                        ForEach(lanes(store.model)) { NDJobLaneView(store: store, lane: $0) }
                    }
                    .padding(.top, 4)
                }
            }
            .padding(.top, fault == nil ? 20 : 18)
        }
    }

    struct Lane: Identifiable {
        var id: String { job.rawValue }
        var job: BrainLayerLaunchdJob
        var name: String
        var enabled: Bool?
        var band: Bool
        var events: [NDRail.Event]
        var olderLabel: String?
        var unknown: Bool
        var number: NDModel.Number
        var streak: Int?
        var fault: Bool
        var accessibility: String
    }

    func lanes(_ m: NDModel) -> [Lane] {
        let now = store.inputs.capture.now
        let failing = store.inputs.capture.maintenanceNightFailures.flatMap { $0 >= 3 ? $0 : nil }
        let alerted = Set(m.faults.flatMap(\.jobs))
        guard let jobs = store.inputs.jobs else {
            return NDStore.jobOrder.map { job in
                Lane(job: job, name: Say.jobNames[job.rawValue] ?? job.title, enabled: nil, band: false, events: [], olderLabel: nil,
                     unknown: true, number: .unknown, streak: nil, fault: false, accessibility: "\(Say.jobNames[job.rawValue] ?? job.title), not known")
            }
        }
        return jobs.map { j in
            let name = Say.jobNames[j.job.rawValue] ?? j.job.title
            let streak = j.job == .maintenanceNightly ? failing : nil
            let fault = streak != nil || alerted.contains(j.job)
            var events: [NDRail.Event] = []
            var older: String? = nil
            if !j.continuous, let last = j.lastRun {
                let h = last.timeIntervalSince(now) / 3600
                // A success dot only for a recorded success (F02); a failure or a counted fault is a Fault dot; anything
                // else (still running, deferred, no status, a damaged history) is the dashed unknown ring.
                let kind: NDRail.Event.Kind = fault || j.lastOutcome == .failed ? .fail : j.lastOutcome == .ok ? .ok : .unknown
                events.append(.init(hours: h, kind: kind))
                if h < -Self.past { older = "\(NDFormat.ago(-h * 3600))" }
                // A failing streak: the earlier failed nights, a day apart, at the lane's left end past the window as a red
                // chevron and "N more" (R1.1 change 25).
                if let streak, streak > 1 {
                    var outside = 0
                    for k in 1..<streak {
                        let earlier = h - Double(k) * 24
                        if earlier >= -Self.past { events.append(.init(hours: earlier, kind: .fail)) } else { outside += 1 }
                    }
                    if outside > 0 {
                        events.append(.init(hours: -Self.past - 1, kind: .fail))
                        older = BrainBarMenuPresenter.SurfaceSay.more(outside)
                    }
                }
            }
            if !j.continuous, let next = j.nextRun, j.enabled != false { events.append(.init(hours: next.timeIntervalSince(now) / 3600, kind: .next)) }
            let number: NDModel.Number
            if let streak {
                number = NDModel.Number(value: String(streak), unit: "nights", tone: .fault, strong: true)  // comp 13
            } else if j.enabled == false {
                number = NDModel.Number(value: BrainBarMenuPresenter.SurfaceSay.off, secondary: true)  // fix 10: Ink 2
            } else if j.continuous {
                number = j.loaded == false ? .unknown : NDModel.Number(value: BrainBarMenuPresenter.SurfaceSay.running, secondary: true)
            } else if let next = j.nextRun, j.enabled != false {
                let a = NDFormat.age(max(0, next.timeIntervalSince(now)))
                number = NDModel.Number(value: "in " + a.value, unit: a.unit)
            } else {
                number = NDModel.Number(value: "")
            }
            let spoken = [name, j.enabled == false ? "off" : nil,
                          streak.map { "failed \($0) nights in a row" },
                          j.continuous ? (j.loaded == false ? "not running" : BrainBarMenuPresenter.SurfaceSay.running) : nil,
                          j.lastRun.map { "last ran \(NDFormat.age(now.timeIntervalSince($0)).spoken) ago" + Self.outcomeWords(j.lastOutcome) },
                          j.nextRun.map { "next run in \(NDFormat.age(max(0, $0.timeIntervalSince(now))).spoken)" }]
                .compactMap { $0 }.joined(separator: ", ")
            return Lane(job: j.job, name: name, enabled: j.enabled, band: j.continuous && j.loaded != false && j.enabled != false,
                        events: events, olderLabel: older, unknown: j.continuous && j.loaded == nil,
                        number: number, streak: streak, fault: fault, accessibility: spoken)
        }
    }
}

/// One job at 24 pt: `220 name | rail | 84 Last run | 84 Next run | 44 On`. The name carries its help tag (§ 4.5).
struct NDJobLaneView: View {
    @ObservedObject var store: NDStore
    let lane: NDJobsPage.Lane

    var body: some View {
        let input = store.inputs.jobs?.first { $0.job == lane.job }
        let now = store.inputs.capture.now
        let frequent = NDJobsPage.frequent.contains(lane.job)
        let last: (String, NDModel.Tone) = {
            if lane.streak != nil { return (Say.failed, .fault) }
            if input?.continuous == true { return input?.loaded == false ? (NDModel.Number.unknown.value, .fine) : (Say.running, .fine) }
            if let at = input?.lastRun { return (Say.ago(NDFormat.ageText(max(0, now.timeIntervalSince(at)))), .fine) }
            return (NDModel.Number.unknown.value, .fine)
        }()
        let next: String = {
            if lane.number.value == Say.off { return Say.off }
            if input?.continuous == true { return NDModel.Number.unknown.value }
            guard let at = input?.nextRun, input?.enabled != false else { return NDModel.Number.unknown.value }
            return Say.inAge(NDFormat.ageText(max(0, at.timeIntervalSince(now))))
        }()
        HStack(spacing: 0) {
            Text(lane.name).font(lane.fault ? NDType.bodyStrong : NDType.body).foregroundStyle(ND.t1)
                .frame(width: NDJobsPage.nameColumn, alignment: .leading)
                .help(Say.jobHelp[lane.job.rawValue] ?? "")
            NDRail(past: NDJobsPage.past, ahead: NDJobsPage.ahead, events: frequent ? [] : lane.events, band: lane.band,
                   dotted: frequent && lane.enabled != false, unknown: lane.unknown, olderLabel: lane.olderLabel)
            Text(last.0).font(last.1 == .fault ? NDType.numberStrong : NDType.number)
                .foregroundStyle(last.1 == .fault ? ND.redText : ND.t2)
                .frame(width: NDJobsPage.valueColumn, alignment: .trailing)
            Text(next).font(NDType.number).foregroundStyle(ND.t2).frame(width: NDJobsPage.valueColumn, alignment: .trailing)
            toggle.frame(width: NDJobsPage.switchColumn, alignment: .trailing)
        }
        .frame(height: NDGrid.row)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(lane.accessibility)
    }

    /// The system switch, mini, in the system accent when on (R1.6 item 12); its help says what a click does.
    @ViewBuilder private var toggle: some View {
        if let enabled = lane.enabled {
            Toggle(lane.name, isOn: Binding(get: { enabled }, set: { store.setJob(lane.job, enabled: $0) }))
                .labelsHidden()
                .toggleStyle(.switch)
                .controlSize(.mini)
                .tint(ND.switchOn)
                .help(enabled ? Say.turnOff(lane.name) : Say.turnOn(lane.name))
        }
    }
}

extension NDJobsPage {
    static func outcomeWords(_ o: NDRunOutcome) -> String {
        switch o {
        case .ok: ""
        case .failed: ", it failed"
        case .running: ", still running"
        case .unknown: ", its result not known"
        }
    }
}

extension NDFormat {
    /// "today at 03:58", "yesterday at 23:40", "Nov 9 at 23:40" (the words the presenter uses for a past time).
    static func day(_ date: Date, now: Date, zone: TimeZone) -> String {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = zone
        let f = DateFormatter()
        f.calendar = calendar
        f.timeZone = zone
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "HH:mm"
        let time = f.string(from: date)
        if calendar.isDate(date, inSameDayAs: now) { return "today at " + time }
        if let yesterday = calendar.date(byAdding: .day, value: -1, to: now), calendar.isDate(date, inSameDayAs: yesterday) {
            return "yesterday at " + time
        }
        f.dateFormat = "MMM d"
        return f.string(from: date) + " at " + time
    }
}


extension String {
    /// "today at 03:16" → "Today at 03:16" (a value that starts its cell).
    var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}
