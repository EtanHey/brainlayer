import SwiftUI

/// The rails' shared grid: `186 name | rail (fills) | 84 number | 40 switch` (§ 2.4).
struct NDRailAxis: View {
    let ticks: [(label: String, hours: Double)]
    let past: Double
    let ahead: Double
    let numberHeader: String
    var limitHours: Double? = nil
    /// Review polish 5: the Jobs page names its switch column ("On"), set like the number column's header.
    var switchHeader: String? = nil

    var body: some View {
        HStack(spacing: 0) {
            Spacer().frame(width: NDGrid.railName)
            GeometryReader { geo in
                let width = geo.size.width
                ForEach(Array(ticks.enumerated()), id: \.offset) { _, tick in
                    let x = CGFloat((tick.hours + past) / (past + ahead)) * width
                    Text(tick.label).font(NDType.axis).foregroundStyle(ND.ink2).fixedSize()
                        .alignmentGuide(.leading) { d in
                            // Centred on its time, kept inside the rail.
                            let left = min(max(x - d.width / 2, 0), width - d.width)
                            return -left
                        }
                }
                .frame(width: width, alignment: .leading)
            }
            Text(numberHeader).font(NDType.axis).foregroundStyle(ND.ink2)
                .frame(width: NDGrid.railNumber, height: 18, alignment: .topTrailing)
            Text(switchHeader ?? "").font(NDType.axis).foregroundStyle(ND.ink2)
                .frame(width: NDGrid.railSwitch, height: 18, alignment: .topTrailing)
        }
        .frame(height: 18)
        .accessibilityHidden(true)
    }
}

// MARK: - Backups

struct NDBackupsPage: View {
    @ObservedObject var store: NDStore
    static let past: Double = 144
    static let ahead: Double = 24
    static let limit: Double = -36

    var body: some View {
        let m = store.model
        VStack(alignment: .leading, spacing: 0) {
            NDPageTop(title: "Every copy, this week", mark: .none, checked: m.checked)
            VStack(spacing: 0) {
                NDRailAxis(ticks: [("6 d ago", -144), ("3 d ago", -72), ("36 h ago", -36), ("now", 0), ("in 24 h", 24)],
                           past: Self.past, ahead: Self.ahead, numberHeader: "newest", limitHours: Self.limit)
                ForEach(lanes(m)) { lane in
                    NDBackupLaneView(lane: lane)
                }
            }
            .overlay(alignment: .topLeading) { limitLine }
            .padding(.top, 14)
            driveGroup(m)
            macGroup(m)
        }
    }

    /// The 36 h limit, through every lane: each newest dot must sit to its right.
    private var limitLine: some View {
        GeometryReader { geo in
            let railWidth = geo.size.width - NDGrid.railName - NDGrid.railNumber - NDGrid.railSwitch
            let x = NDGrid.railName + CGFloat((Self.limit + Self.past) / (Self.past + Self.ahead)) * railWidth
            ND.track.frame(width: 1, height: max(0, geo.size.height - 22)).offset(x: x, y: 18)
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    struct Lane: Identifiable {
        var id: String
        var name: String
        var events: [NDRail.Event]
        var unknown: Bool
        var stalledFrom: Double?
        /// Review N1: set when the newest cell is unknown; the rail claims no confirmed copy.
        var unconfirmedFrom: Double? = nil
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
            var events = (attempts ?? []).map { NDRail.Event(hours: $0.at.timeIntervalSince(now) / 3600, kind: $0.ok ? .ok : .fail) }
                .filter { $0.hours >= -Self.past }
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
                        number: number, strong: row?.strong ?? false, accessibility: spoken + newest)
        }
        return [
            lane("backup_local", "Copy on this Mac", lanes?.local, next: databaseNext, stalls: false),
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

    @ViewBuilder private func driveGroup(_ m: NDModel) -> some View {
        let row = m.areas.first { $0.thing == .safe }?.rows.first { $0.id == "drive_auth" }
        let fault = row?.number.tone == .fault
        let missing = store.inputs.capture.driveAuthModel == .missing
        let state: String = switch row?.number.value {
        case "connected"?: "Connected"
        case "disconnected"?: "Disconnected"
        case "not connected"?: "Not connected"
        default: "—"
        }
        let fact = fault ? (missing ? BrainBarMenuPresenter.Say.s13DetailMissing.text : BrainBarMenuPresenter.Say.s13Detail.text)
            : state == "Connected" ? BrainBarMenuPresenter.SurfaceSay.driveUploadsHere : ""
        if state == "—", !fault {
            // B3: an unknown Drive state is one row, its name and its state cell, as every other unknown is drawn.
            NDFactRow(name: "Google Drive", fact: state) { EmptyView() }.padding(.top, 18)
        } else {
        NDGroupLabel(title: "Google Drive", fault: fault)
        NDFactRow(name: state, nameTone: fault ? .fault : .fine, fact: fact) {
            if fault {
                let label = missing ? BrainBarMenuPresenter.Say.connect.text : BrainBarMenuPresenter.Say.reconnect.text
                NDWaitingSwap(waiting: store.inputs.driveReconnecting) {
                    NDActionButton(label: label, large: true) { store.reconnectDrive() }
                } line: {
                    HStack(spacing: 7) {
                        NDMarkView(mark: .running)
                        Text(BrainBarMenuPresenter.SurfaceSay.waitingForBrowser).font(NDType.fact).foregroundStyle(ND.ink2)
                    }
                }
            }
        }
        }
    }

    /// B3: "Newest copy" follows the "Copy on this Mac" lane: one reading, one staleness rule. When the lane's age is
    /// unknown (a stale or missing check), the time is unknown too.
    func newestCopyFact(_ m: NDModel) -> String {
        let lane = lanes(m).first { $0.id == "backup_local" }
        guard let at = store.inputs.newestLocalCopyAt, let lane, lane.number.value != NDModel.Number.unknown.value else { return "—" }
        return NDFormat.day(at, now: store.inputs.capture.now, zone: store.inputs.capture.timeZone)
    }

    @ViewBuilder private func macGroup(_ m: NDModel) -> some View {
        let disk = m.areas.first { $0.thing == .safe }?.rows.first { $0.id == "disk" }
        NDGroupLabel(title: "On this Mac", fault: false)
        NDFactRow(name: "Newest copy", fact: newestCopyFact(m)) {
            if store.inputs.canRevealLocalCopy {
                Button("Show in Finder") { store.revealLocalCopy() }.buttonStyle(NDLinkButtonStyle())
            }
        }
        NDFactRow(name: "Room for backups", fact: disk?.number.value ?? "—", factTone: disk?.number.tone ?? .fine) { EmptyView() }
    }
}

struct NDBackupLaneView: View {
    let lane: NDBackupsPage.Lane
    var body: some View {
        HStack(spacing: 0) {
            Text(lane.name).font(lane.strong ? NDType.rowStrong : NDType.row).frame(width: NDGrid.railName, alignment: .leading)
            NDRail(past: NDBackupsPage.past, ahead: NDBackupsPage.ahead, events: lane.events, unknown: lane.unknown,
                   stalledFrom: lane.stalledFrom, unconfirmedFrom: lane.unconfirmedFrom)
            NDNumberText(number: lane.number).frame(width: NDGrid.railNumber, alignment: .trailing)
            Spacer().frame(width: NDGrid.railSwitch)
        }
        .frame(height: 52)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(lane.accessibility)
    }
}

struct NDGroupLabel: View {
    let title: String
    let fault: Bool
    var body: some View {
        Text(title).font(NDType.areaName).tracking(0.2).foregroundStyle(fault ? ND.fault : ND.ink)
            .padding(.top, 18).padding(.bottom, 6)
            .accessibilityAddTraits(.isHeader)
    }
}

struct NDFactRow<Action: View>: View {
    let name: String
    var nameTone: NDModel.Tone = .fine
    let fact: String
    var factTone: NDModel.Tone = .fine
    @ViewBuilder var action: Action

    var body: some View {
        HStack(spacing: 0) {
            Text(name).font(nameTone == .fault ? NDType.rowStrong : NDType.row)
                .foregroundStyle(nameTone == .fault ? ND.fault : ND.ink)
                .frame(width: NDGrid.railName, alignment: .leading)
            Text(fact).font(NDType.row)
                .foregroundStyle(factTone == .fault ? ND.fault : factTone == .warn ? ND.warn : ND.ink2)
                .frame(maxWidth: .infinity, alignment: .leading)
            action
        }
        .frame(minHeight: 30)
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

// MARK: - Jobs

struct NDJobsPage: View {
    @ObservedObject var store: NDStore
    static let past: Double = 36
    static let ahead: Double = 12

    static let names: [BrainLayerLaunchdJob: String] = [
        .watch: "Read new conversations", .hotlane: "Keep search ready", .drain: "Save queued memories",
        .walCheckpoint: "Tidy up storage", .backupDaily: "Back up the database", .jsonlBackup: "Back up conversations",
        .maintenanceNightly: "Nightly upkeep", .maintenanceWeekly: "Weekly upkeep", .decay: "Fade old memories",
        .repairFTS: "Repair word search",
    ]

    var body: some View {
        let m = store.model
        VStack(alignment: .leading, spacing: 0) {
            NDPageTop(title: "Background jobs", mark: .none, checked: m.checked)
            VStack(spacing: 0) {
                NDRailAxis(ticks: [("36 h ago", -36), ("24 h ago", -24), ("12 h ago", -12), ("now", 0), ("in 12 h", 12)],
                           past: Self.past, ahead: Self.ahead, numberHeader: "next run",
                           switchHeader: BrainBarMenuPresenter.SurfaceSay.onHeader)
                ForEach(lanes(m)) { NDJobLaneView(store: store, lane: $0) }
            }
            .padding(.top, 14)
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
                Lane(job: job, name: Self.names[job] ?? job.title, enabled: nil, band: false, events: [], olderLabel: nil,
                     unknown: true, number: .unknown, streak: nil, fault: false, accessibility: "\(Self.names[job] ?? job.title), not known")
            }
        }
        return jobs.map { j in
            let name = Self.names[j.job] ?? j.job.title
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

struct NDJobLaneView: View {
    @ObservedObject var store: NDStore
    let lane: NDJobsPage.Lane

    var body: some View {
        VStack(spacing: 6) {
            HStack(spacing: 0) {
                Text(lane.name).font(lane.fault ? NDType.rowStrong : NDType.row).frame(width: NDGrid.railName, alignment: .leading)
                NDRail(past: NDJobsPage.past, ahead: NDJobsPage.ahead, events: lane.events, band: lane.band, unknown: lane.unknown,
                       olderLabel: lane.olderLabel.map { $0 })
                HStack(spacing: 6) {
                    if let streak = lane.streak { NDStreakDots(count: streak) }
                    NDNumberText(number: lane.number)
                }
                .frame(width: NDGrid.railNumber + (lane.streak == nil ? 0 : 24), alignment: .trailing)
                .padding(.leading, lane.streak == nil ? 0 : -24)
                toggle.frame(width: NDGrid.railSwitch, alignment: .trailing)
            }
            .frame(height: lane.fault ? 24 : 40)
            if lane.fault {
                HStack {
                    Spacer()
                    Button(BrainBarMenuPresenter.Say.copyReport.text) { store.copyReport() }.buttonStyle(NDLinkButtonStyle())
                }
            }
        }
        .padding(.top, lane.fault ? 13 : 0)
        .frame(height: lane.fault ? 58 + 10 : 40, alignment: .top)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(lane.accessibility)
    }

    @ViewBuilder private var toggle: some View {
        if let enabled = lane.enabled {
            Toggle(lane.name, isOn: Binding(get: { enabled }, set: { store.setJob(lane.job, enabled: $0) }))
                .labelsHidden()
                .toggleStyle(NDSwitchStyle())
                .help(enabled ? BrainBarMenuPresenter.SurfaceSay.turnOff(lane.name) : BrainBarMenuPresenter.SurfaceSay.turnOn(lane.name))
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

/// The Jobs switch as the comps draw it (10, 13): a 26 × 15 pt capsule, Ink 2 when on and Track when off, a 13 pt white
/// knob. The system switch's on-state read as off in a dark window and near-black in a light one (design pre-pass 3).
/// It stays a `Toggle`, so VoiceOver reads a switch with its value; Space toggles it when focused.
struct NDSwitchStyle: ToggleStyle {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func makeBody(configuration: Configuration) -> some View {
        Button { configuration.isOn.toggle() } label: {
            Capsule(style: .continuous)
                .fill(configuration.isOn ? ND.ink2 : ND.track)
                .frame(width: 26, height: 15)
                .overlay(alignment: configuration.isOn ? .trailing : .leading) {
                    Circle().fill(Color.white).frame(width: 13, height: 13)
                        .shadow(color: .black.opacity(0.3), radius: 0.75, y: 0.5)
                        .padding(1)
                }
                .animation(reduceMotion ? nil : .easeOut(duration: 0.15), value: configuration.isOn)
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityRepresentation { Toggle(isOn: configuration.$isOn) { configuration.label } }
    }
}
