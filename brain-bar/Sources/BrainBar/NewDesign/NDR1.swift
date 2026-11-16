import Foundation

/// What round 1 of the design (DESIGN-SPEC-R1) draws on top of `NDModel`: the headline items with their pipe strips and
/// one action each, the dropdown's four rows, the window footer and the pipeline. Pure: one value from `NDInputs` and
/// the `NDModel` made from them. It never says less than the presenter: every presenter fault the pipeline does not
/// explain is kept as its own item, in the presenter's words.
struct NDR1: Equatable, Sendable {
    typealias Say = BrainBarMenuPresenter.SurfaceSay
    typealias P = BrainBarMenuPresentation

    /// The broken pipe cut to its two stages, for the dropdown (§ 3.2, "In the dropdown").
    struct Strip: Equatable, Sendable {
        var from: String
        var to: String
        var side: NDBreakSide
        var severity: NDSeverity
    }

    struct Item: Equatable, Sendable, Identifiable {
        var id: String
        var severity: NDSeverity
        var title: String
        var detail: String?
        var action: P.Action?
        var strip: Strip?
        /// The dropdown's shorter title, when the deck gives one ("Nightly check-up failed 3 nights").
        var menuTitle: String? = nil
        /// The dropdown's one line under the title (R1.1 change 28; the strips are gone).
        var menuDetail: String? = nil
        /// The page that owns it (the dropdown's row values turn red when concerned).
        var concerns: Set<String>
    }

    struct MenuRow: Equatable, Sendable, Identifiable {
        enum Kind: String, Sendable { case activity, search, backups, jobs }
        var kind: Kind
        var id: String { kind.rawValue }
        var label: String
        var value: String
        var tone: NDModel.Tone
        var help: String?
    }

    var pipeline: NDPipelineState
    /// "All good", one item's title, "N things need you", or "Not confirmed".
    var headline: String
    var headlineSymbol: String?
    var headlineTone: NDModel.Tone
    var items: [Item]
    /// Not confirmed: the stale verdict's time and age.
    var unconfirmedSince: String?
    var unconfirmedDetail: String?
    var menuRows: [MenuRow]
    var footerLeft: String
    var footerRight: String?
    var calm: Bool { items.isEmpty && unconfirmedSince == nil }
    var redCount: Int { items.filter { $0.severity == .red }.count }
    /// Tab badges (D6): red items only, by the page that owns them.
    var backupsBadge: Int {
        items.filter { $0.severity == .red && !$0.concerns.isDisjoint(with: NDR1.backupChecks) }.count
    }
    var jobsBadge: Int {
        items.filter { $0.severity == .red && ($0.id == "nightly" || $0.concerns.contains("nightly")) }.count
    }
    static let backupChecks: Set<String> = ["drive_auth", "backup_local", "backup_offsite", "transcripts", "integrity", "disk"]

    static func make(_ i: NDInputs, _ m: NDModel) -> NDR1 {
        let c = i.capture
        let pipeline = NDPipelineState.make(i)
        let verdict = c.verdict
        func check(_ name: String) -> BrainBarHealthVerdict.Check? { verdict?.checks.first { $0.name == name } }
        var items: [Item] = []
        var covered: Set<String> = []
        let copyReport = P.Action(kind: .copyReport, label: BrainBarMenuPresenter.Say.copyReport.text)

        if !pipeline.allUnconfirmed {
            // Google Drive refuses: one cause, two connections (c10, c11).
            if pipeline.faultStations.contains(.drive) {
                let action = m.faults.first { $0.action.map { [.reconnectDrive, .connectDrive].contains($0.kind) } == true }?.action
                    ?? P.Action(kind: .reconnectDrive, label: BrainBarMenuPresenter.Say.reconnect.text)
                items.append(Item(id: "drive", severity: .red, title: Say.driveSignedOut,
                                  detail: c.driveAuthModel == .missing ? BrainBarMenuPresenter.Say.s13DetailMissing.text
                                      : BrainBarMenuPresenter.Say.s13Detail.text,
                                  action: action, strip: Strip(from: Say.stageMac, to: Say.stageDrive, side: .target, severity: .red),
                                  menuDetail: driveAges(verdict).min().map { Say.nothingUploadedFor(NDFormat.ageText($0)) },
                                  concerns: ["drive_auth", "backup_offsite", "transcripts"]))
                covered.formUnion(["drive_auth", "backup_offsite", "transcripts"])
            }
            // One agent's sessions not read, or the reader itself.
            if let ingest = check("ingest"), let sev = sev(ingest) {
                let agents = NDPipelineState.agentCodes(ingest)
                if let first = agents.first {
                    items.append(Item(id: "agent", severity: sev, title: Say.agentNotRead(first.label), detail: Say.agentNotReadDetail,
                                      action: sev == .red ? copyReport : nil, strip: Strip(from: first.label, to: Say.stageRead, side: .source, severity: sev),
                                      concerns: ["ingest"]))
                } else {
                    // D6: amber is late, not stopped: "Catching up on new conversations", no "needs you", no button.
                    items.append(Item(id: "reader", severity: sev,
                                      title: sev == .red ? Say.readerNotReading : BrainBarMenuPresenter.Say.r02Headline.text,
                                      detail: sev == .red ? BrainBarMenuPresenter.Say.cantFix.text : nil, action: sev == .red ? copyReport : nil,
                                      strip: Strip(from: Say.goesIn, to: Say.stageRead, side: .target, severity: sev), concerns: ["ingest"]))
                }
                covered.insert("ingest")
            }
            // Saves stuck in the queue (c4, target).
            if let stores = check("stores"), let sev = sev(stores), let queued = i.oldestSaveQueuedAt {
                let age = NDFormat.ageText(max(0, c.now.timeIntervalSince(queued)))
                items.append(Item(id: "saves", severity: sev, title: Say.savesStuck,
                                  detail: Say.savesWaited(i.savesWaiting.flatMap { $0 > 0 ? $0 : nil }, age), action: sev == .red ? copyReport : nil,
                                  strip: Strip(from: Say.stageQueue, to: Say.stageMemory, side: .target, severity: sev), concerns: ["stores"]))
                covered.insert("stores")
            }
            // The nightly check-up failing (c12, source).
            if let nights = c.maintenanceNightFailures, nights >= 3 {
                let since = lastNightly(i).map { Say.notCheckedSince(weekday($0, c)) }
                items.append(Item(id: "nightly", severity: .red, title: Say.nightlyFailed(nights),
                                  detail: since ?? BrainBarMenuPresenter.Say.cantFix.text, action: copyReport,
                                  strip: Strip(from: Say.jobs, to: Say.stageMemory, side: .source, severity: .red),
                                  menuTitle: Say.nightlyFailedShort(nights), menuDetail: since,
                                  concerns: ["maintenanceNightly", "nightly"]))
                covered.formUnion(["maintenanceNightly", "nightly"])
            }
        }
        // Search by meaning behind is drawn below as its own (orange or red) item.
        if !pipeline.allUnconfirmed, check("embedding").flatMap(sev) != nil { covered.insert("embedding") }
        // Every presenter fault the pipeline did not explain, in the presenter's own words.
        for f in m.faults {
            let explained = !f.concerns.isDisjoint(with: covered)
                || (covered.contains("maintenanceNightly") && (f.jobs.contains(.maintenanceNightly) || f.thing == .upkeep))
                || (covered.contains("embedding") && f.thing == .finding)
            guard !explained else { continue }
            if f.thing == .finding, check("embedding").flatMap(sev) != nil { continue }  // drawn as "behind" below
            // D6: severity comes from the verdict: red only when a check it concerns is red (or it is not a check's: a job
            // alert or a runtime warning); a fault over amber checks only is late, so amber, and no report button.
            let concerned = (verdict?.checks ?? []).filter { f.concerns.contains($0.name) }
            let red = concerned.isEmpty || concerned.contains { $0.state == .red }
            let action = red ? f.action : f.action.flatMap { NDButtonPolicy.filled($0) ? $0 : nil }
            items.append(Item(id: "p." + f.title, severity: red ? .red : .orange, title: f.title, detail: f.shownFacts.joined(separator: " "),
                              action: action, strip: nil, concerns: f.concerns))
        }
        // Search by meaning behind (orange, late) or stalled (red): no button, there is nothing to do.
        if !pipeline.allUnconfirmed, let embedding = check("embedding"), let s = sev(embedding) {
            let age = NDFormat.ageText(max(0, embedding.ageS ?? 0))
            items.append(Item(id: "meaning", severity: s, title: s == .red ? Say.meaningBehind(age) : Say.meaningCatchingUp,
                              detail: Say.meaningBehindDetail, action: nil,
                              strip: Strip(from: Say.stageMemory, to: Say.stageMeaning, side: .target, severity: s), concerns: ["embedding"]))
        }
        // One filled button per surface: the first item's action; the copy-report button is never the filled one when a
        // reconnect exists (kept as data: the views read `filledActionIndex`).

        // Headline.
        var unconfirmedSince: String?
        var unconfirmedDetail: String?
        var headline = BrainBarMenuPresenter.Say.allGood.text
        var symbol: String?
        var tone: NDModel.Tone = .fine
        if pipeline.allUnconfirmed, items.isEmpty {
            // R1.1 change 32: said once, in the title; the detail says nothing is known to be broken.
            headline = Say.notConfirmed
            symbol = "questionmark.circle"
            if let at = verdict?.evaluatedAt {
                unconfirmedSince = Say.notConfirmedSince(hhmm(at, c))
                headline = unconfirmedSince ?? headline
                unconfirmedDetail = Say.nothingKnownBroken(NDFormat.ageText(max(0, c.now.timeIntervalSince(at))))
            }
        } else if items.count == 1, let only = items.first {
            headline = only.title
            symbol = only.severity == .red ? "exclamationmark.circle.fill" : "exclamationmark.triangle.fill"
            tone = only.severity == .red ? .fault : .warn
        } else if items.count > 1 {
            // D6: "needs you" and its count are for red items only; amber ones are listed, never counted.
            let red = items.filter { $0.severity == .red }.count
            if red == 0, let first = items.first {
                headline = first.title
                symbol = "exclamationmark.triangle.fill"
                tone = .warn
            } else {
                headline = red == 1 ? BrainBarMenuPresenter.Say.countOne.text : Say.thingsNeedYou(red)
                symbol = "exclamationmark.circle.fill"
                tone = .fault
            }
        }

        // Footer (finding 8).
        var left = "", right: String?
        if let at = verdict?.evaluatedAt {
            let age = NDFormat.ageText(max(60, c.now.timeIntervalSince(at)))
            if pipeline.allUnconfirmed {
                left = Say.waitingForNextCheck
                _ = age
            } else {
                left = Say.checkedAgo(age)
                let next = at.addingTimeInterval(300).timeIntervalSince(c.now)
                if next > 0 { right = Say.nextCheckIn(NDFormat.ageText(max(60, next))) }
            }
        }

        return NDR1(pipeline: pipeline, headline: headline, headlineSymbol: symbol, headlineTone: tone, items: items,
                    unconfirmedSince: unconfirmedSince, unconfirmedDetail: unconfirmedDetail,
                    menuRows: menuRows(i, m, items: items, pipeline: pipeline), footerLeft: left, footerRight: right)
    }

    /// The Drive copies' ages (database and conversations), measured ones only.
    static func driveAges(_ verdict: BrainBarHealthVerdict?) -> [Double] {
        (verdict?.checks ?? []).filter { ["backup_offsite", "transcripts"].contains($0.name) }
            .compactMap(\.ageS).filter { $0.isFinite && $0 >= 0 }
    }

    /// The last nightly check-up that ran (the maintenance run record).
    static func lastNightly(_ i: NDInputs) -> Date? {
        if case let .ran(at) = i.nightlyLastRun { return at }
        return nil
    }

    /// "Thursday": the day a past date fell on, in the capture's zone.
    static func weekday(_ date: Date, _ c: BrainBarMenuCapture) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = c.timeZone
        f.dateFormat = "EEEE"
        return f.string(from: date)
    }

    private static func sev(_ k: BrainBarHealthVerdict.Check) -> NDSeverity? {
        switch k.state {
        case .red: .red
        case .amber: .orange
        default: nil
        }
    }

    static func hhmm(_ date: Date, _ c: BrainBarMenuCapture) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = c.timeZone
        f.dateFormat = "HH:mm"
        return f.string(from: date)
    }

    /// The dropdown's four rows (§ 2.2): what each area is doing, in one value.
    static func menuRows(_ i: NDInputs, _ m: NDModel, items: [Item], pipeline: NDPipelineState) -> [MenuRow] {
        let c = i.capture
        let fresh = !pipeline.allUnconfirmed
        let checks = c.verdict?.checks ?? []
        // Activity: the last hour, as the help tag says it.
        let hour = i.menuWave
        let activityHelp = hour.map { w in
            Say.lastHour(saved: NDFormat.integer(w.savesTotal), read: NDFormat.integer(w.readsTotal))
        }
        // D2: a row may show a fine value only when its stages read fine on the Overview; otherwise it leads with what is
        // wrong (the stage's status words), or "—" when nothing is confirmed.
        func worst(_ stations: [NDStation]) -> NDPipelineState.Condition {
            let all = stations.map(pipeline.condition)
            if let red = all.first(where: { if case .issue(_, .red) = $0 { true } else { false } }) { return red }
            if let issue = all.first(where: { if case .issue = $0 { true } else { false } }) { return issue }
            return all.contains(.notConfirmed) ? .notConfirmed : .fine
        }
        func tone(_ s: NDSeverity) -> NDModel.Tone { s == .red ? .fault : .warn }
        // Search: the share searchable by meaning, when search by meaning and search read fine.
        let percent = i.searchable.map { Int(($0 * 100).rounded(.down)) }
        var searchValue = percent.map { Say.percentReady($0) } ?? NDModel.Number.unknown.value
        var searchTone: NDModel.Tone = .fine
        switch worst([.meaning, .search]) {
        case .fine: break
        case .notConfirmed: searchValue = NDModel.Number.unknown.value
        case let .issue(words, sev):
            searchValue = words ?? (sev == .red ? Say.failed : Say.late)
            searchTone = tone(sev)
        }
        let searchHelp = i.searchable.map { Say.searchable(String(format: "%.1f", $0 * 100)) }
        // Backups: the oldest of the three newest copies, so the number is the one closest to late.
        let copyNames = ["backup_local", "backup_offsite", "transcripts"]
        let ages = copyNames.compactMap { name in checks.first { $0.name == name }?.ageS }.filter { $0.isFinite && $0 >= 0 }
        let backupsConcerned = items.contains { !$0.concerns.isDisjoint(with: Set(copyNames + ["drive_auth", "integrity", "disk"])) }
        var backupsValue: String
        var backupsTone: NDModel.Tone = backupsConcerned ? .fault : .fine
        if fresh, ages.count == copyNames.count, let oldest = ages.max() {
            backupsValue = Say.ago(NDFormat.ageText(oldest))
        } else {
            backupsValue = NDModel.Number.unknown.value
        }
        switch worst([.mac, .drive]) {
        case .fine: break
        case .notConfirmed: backupsValue = NDModel.Number.unknown.value
        case let .issue(_, sev):
            if backupsTone == .fine { backupsTone = tone(sev) }
            // D7.3: name the copy that is stale, with one number for one fact. Drive: the last time anything reached it
            // (the same age as "Nothing has uploaded for …"); otherwise the copy on this Mac.
            if pipeline.condition(.drive) != .fine, let newest = driveAges(c.verdict).min() {
                backupsValue = Say.driveAgo(NDFormat.ageText(newest))
            } else if let mac = checks.first(where: { $0.name == "backup_local" })?.ageS, mac.isFinite, mac >= 0 {
                backupsValue = Say.macAgo(NDFormat.ageText(mac))
            }
        }
        let backupsHelp: String? = fresh && ages.count == copyNames.count ? zip([Say.laneMac, Say.laneDrive, Say.laneConversations], ages)
            .map { "\($0.0): " + Say.ago(NDFormat.ageText($0.1)) }.joined(separator: ", ") : nil
        // Jobs: fine, failed (red) or late (orange).
        let jobsFailed = items.contains { $0.id == "nightly" } || m.faults.contains { !$0.jobs.isEmpty || $0.thing == .upkeep }
        let upkeep = m.areas.first { $0.thing == .upkeep }
        let jobsLate = upkeep?.rows.contains { $0.number.tone == .warn } == true
        let jobsValue: String
        let jobsTone: NDModel.Tone
        let jobsStage = worst([.jobs])
        if !fresh || jobsStage == .notConfirmed {
            jobsValue = NDModel.Number.unknown.value
            jobsTone = .fine
        } else if case let .issue(_, sev) = jobsStage {
            jobsValue = sev == .red ? Say.failed : Say.late
            jobsTone = tone(sev)
        } else if jobsFailed {
            jobsValue = Say.failed
            jobsTone = .fault
        } else if jobsLate {
            jobsValue = Say.late
            jobsTone = .warn
        } else {
            jobsValue = Say.fine
            jobsTone = .fine
        }
        let nextJob = (i.jobs ?? []).compactMap { j -> Date? in j.enabled == false ? nil : j.nextRun }.filter { $0 > c.now }.min()
        let jobsHelp = nextJob.map { Say.nextRun + " " + Say.inAge(NDFormat.ageText($0.timeIntervalSince(c.now))) }
        return [
            MenuRow(kind: .activity, label: Say.activity, value: "", tone: .fine, help: activityHelp),
            MenuRow(kind: .search, label: Say.search, value: searchValue, tone: searchTone, help: searchHelp),
            MenuRow(kind: .backups, label: Say.backups, value: backupsValue, tone: backupsTone, help: backupsHelp),
            MenuRow(kind: .jobs, label: Say.jobs, value: jobsValue, tone: jobsTone, help: jobsHelp),
        ]
    }
}
