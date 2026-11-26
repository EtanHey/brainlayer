import Foundation

/// The closed list (phase 8, P2b): every sentence, headline, label and standing item the presenter can put in its value,
/// each with a stable id (its truth-table row or contract case), its text or template, a kind, and, for a claim or a
/// reassurance, the condition on the capture under which it is true. The presenter builds every string from these entries
/// and the word formatters below; `finalGuard` checks every sentence it is about to return against them. The invariant
/// test keeps its own conditions under the same ids and fails on any emitted sentence that is not registered here.
extension BrainBarMenuPresenter {
    /// I3/R30: which things a reassurance covers, using the same nouns and uncovered-row rule as its condition.
    static func reassuranceCoverage(_ id: String, _ values: [String], _ p: P, _ c: Capture) -> Set<Thing>? {
        switch id {
        case "S1": return Set(four)
        case "A2.fine.is", "A2.fine.are": return Say.fineThings(values[0])
        case "M5": return uncovered(p, c)
        case "U03.stillSaved": return [.remembering, .safe]
        case "F07.exactWords": return [.finding]
        default: return nil
        }
    }

    struct Sentence: Sendable {
        /// A claim states a fact about the world; a reassurance says something is fine; a neutral entry is an instruction,
        /// a label, or a statement of uncertainty, and cannot be false.
        enum Kind: Sendable { case claim, reassurance, neutral }
        typealias Holds = @Sendable (Capture, P, [String]) -> Bool
        let id: String, kind: Kind, template: String
        /// The thing a row whose first sentence is this one speaks for ("Everything else is fine." covers the rest).
        let thing: Thing?
        let holds: Holds?
        private let parts: [String], slots: [String]

        init(_ id: String, _ kind: Kind, _ template: String, _ thing: Thing? = nil, holds: Holds? = nil) {
            (self.id, self.kind, self.template, self.thing, self.holds) = (id, kind, template, thing, holds)
            var parts = [""], slots: [String] = []
            for piece in template.split(separator: "{", omittingEmptySubsequences: false).enumerated() {
                guard piece.offset > 0, let close = piece.element.firstIndex(of: "}") else { parts[parts.count - 1] += piece.element; continue }
                slots.append(String(piece.element[..<close]))
                parts.append(String(piece.element[piece.element.index(after: close)...]))
            }
            (self.parts, self.slots) = (parts, slots)
        }

        var text: String { template }
        func callAsFunction(_ values: String...) -> String { fill(values) }
        func fill(_ values: [String]) -> String { zip(values, parts.dropFirst()).reduce(parts[0]) { $0 + $1.0 + $1.1 } }

        /// The placeholder values when `sentence` is this entry with each slot filled by a word its formatter makes for `c`.
        func match(_ sentence: String, _ c: Capture) -> [String]? {
            guard !slots.isEmpty else { return sentence == template ? [] : nil }
            guard sentence.hasPrefix(parts[0]) else { return nil }
            var rest = Substring(sentence.dropFirst(parts[0].count)), values: [String] = []
            for (i, slot) in slots.enumerated() {
                let next = parts[i + 1], value: Substring
                if i == slots.count - 1 {
                    guard rest.hasSuffix(next) else { return nil }
                    (value, rest) = (rest.dropLast(next.count), "")
                } else {
                    guard let r = rest.range(of: next) else { return nil }
                    (value, rest) = (rest[..<r.lowerBound], rest[r.upperBound...])
                }
                // No slot takes an empty value but `{over}`, whose empty form is the formatter's own (CI on #1790).
                guard !value.isEmpty || slot == "over", Say.fits(slot, String(value), c) else { return nil }
                values.append(String(value))
            }
            return values
        }
    }

    /// The entry and placeholder values for one sentence, or nil when no entry makes it (a fault the guard corrects).
    static func entry(for sentence: String, _ c: Capture) -> (Sentence, [String])? {
        if let exact = Say.exact[sentence] { return (exact, []) }
        for entry in Say.templated { if let values = entry.match(sentence, c) { return (entry, values) } }
        return nil
    }

    enum Say {
        typealias S = Sentence
        private static func red(_ c: Capture, _ name: String) -> Bool { c.verdict?.checks.contains { $0.name == name && $0.state == .red } == true }
        private static func drive(_ c: Capture, missing: Bool) -> Bool { red(c, "drive_auth") && (c.driveAuthModel == .missing) == missing }
        /// F07's truth: an `embedding` check red, or amber within its threshold (above it, progress was seen: F06).
        private static func stalled(_ c: Capture, _ ok: (BrainBarHealthVerdict.Check) -> Bool = { _ in true }) -> Bool {
            c.verdict?.checks.contains { k in
                k.name == "embedding" && (k.state == .red || (k.state == .amber && valid(k.ageS) && k.thresholdS.map { k.ageS! <= $0 } == true)) && ok(k)
            } == true
        }
        /// R02: green or amber ingest lag under the app's escalation limit is being handled.
        static func catchingUp(_ c: Capture) -> Bool { ingestLagDisposition(c) == .catchingUp }
        private static func behind(_ noun: String, _ c: Capture) -> Bool {
            noun == behindNoun[.remembering] ? catchingUp(c) : noun == behindNoun[.safe] && runningUpload(c) != nil
        }
        private static func progress(_ c: Capture, _ value: String) -> Bool { runningUpload(c)?.onSchedule?.progress.map { String(percent($0)) == value } == true }
        /// A copy age at `now`: the newest valid copy among `names`, in the words `copyAge` gives it.
        private static func age(_ c: Capture, _ names: [String], _ value: String) -> Bool {
            guard let v = c.verdict, let age = v.checks.filter({ names.contains($0.name) }).compactMap(\.ageS).filter({ valid($0) }).min() else { return false }
            return copyAge(age, c, v) == value
        }
        /// R25: no copy in Drive of either kind has a verified record (the database copy is present without an age, and no
        /// conversation backup carries one).
        static func noDriveClock(_ c: Capture) -> Bool {
            let offsite = c.verdict?.checks.filter { $0.name == "backup_offsite" } ?? []
            return !offsite.isEmpty && (c.verdict?.checks ?? []).allSatisfy { !["backup_offsite", "transcripts"].contains($0.name) || $0.ageS == nil }
        }
        /// R23: the running upload's start, said as a bare clock time today, and with its day words on another day.
        private static func started(_ c: Capture, today: Bool, _ value: String) -> Bool {
            uploadStart(c).map { $0.today == today && (today ? hhmm($0.date, c) : at($0.date, c)) == value } == true
        }
        private static func countIs(_ p: P, _ value: String) -> Bool { String(p.rows.filter { $0.kind == .needsYou }.count) == value }

        // MARK: R1.11 conditions
        static func states(_ c: Capture, _ name: String, _ states: [State]) -> [BrainBarHealthVerdict.Check] {
            c.verdict?.checks.filter { $0.name == name && states.contains($0.state) } ?? []
        }
        private static func amber(_ c: Capture, _ name: String) -> Bool { !states(c, name, [.amber]).isEmpty }
        /// A watcher alert on a readable, fresh file.
        static func fresh(_ c: Capture, _ reason: String) -> Bool {
            c.watcher.map { $0.readable && $0.fresh && $0.alertReasons.contains(reason) } == true
        }
        /// A7: the provider a `growth_not_indexed:{provider}` alert names, as L38 names it; nil for any other alert.
        static func provider(_ reason: String) -> String? {
            let prefix = "growth_not_indexed:"
            return reason.hasPrefix(prefix) ? providerNames[String(reason.dropFirst(prefix.count))] : nil
        }
        /// A5: a failed database probe makes `coverage_drop` unprovable.
        static func coverageDrop(_ c: Capture) -> Bool { fresh(c, "coverage_drop") && c.watcher?.dbProbeFailed == false }
        /// A7: the providers the census names (`growth_not_indexed:{provider}`, on a failing or late `ingest` check's context, as
        /// the window reads it, or as a fresh watcher alert), as L38 names them; an unknown one is never guessed.
        static func providers(_ c: Capture) -> [String] {
            let codes = states(c, "ingest", [.red, .amber]).flatMap { $0.contextCodes ?? [] }
            let alerts = c.watcher.map { $0.readable && $0.fresh ? $0.alertReasons : [] } ?? []
            var names: [String] = []
            for name in (codes + alerts).compactMap(provider) where !names.contains(name) { names.append(name) }
            return names
        }
        static let providerNames = ["claude": "Claude", "codex": "Codex", "cursor": "Cursor", "gemini": "Antigravity"]
        /// C2, C3: Drive is signed in, so a late Drive copy is not a Reconnect.
        static func driveSignedIn(_ c: Capture) -> Bool {
            let drive = c.verdict?.checks.filter { $0.name == "drive_auth" } ?? []
            return !drive.isEmpty && drive.allSatisfy { $0.state == .green }
        }
        /// D2: the health check itself is late.
        static func monitorLate(_ c: Capture) -> Bool { !states(c, "monitor", [.red, .amber]).isEmpty }
        /// R1.13 (#1827): the producer measures the lag at a new low over its window, with an estimate (`offset_lag` is
        /// then not an alert reason).
        static func producerCatchingUp(_ c: Capture) -> Bool {
            c.watcher.map { $0.readable && $0.fresh && $0.catchUp?.state == "catching_up" } == true
        }
        /// R1.13 (#1827): the producer says the lag stopped shrinking (`catch_up.state` stalled, `offset_lag` raised again).
        static func producerStalled(_ c: Capture) -> Bool { fresh(c, "offset_lag") && c.watcher?.catchUp?.state == "stalled" }
        /// A1: lag on a readable, fresh watcher file, past the app’s 30-minute clock or with `ingest` red; R1.13: or the
        /// producer’s catch-up, catching up or stalled.
        static func lagShown(_ c: Capture) -> Bool {
            (fresh(c, "offset_lag") && (lagPastLimit(c) || red(c, "ingest") || producerStalled(c))) || producerCatchingUp(c)
        }
        /// A1’s “catching up by itself”. R1.13: from a producer with `catch_up`, its own measure (catching_up); from an
        /// older one, L55’s approximation: the lag is the watcher’s only alert (R10’s quarantined records aside).
        static func lagDraining(_ c: Capture) -> Bool {
            guard c.watcher?.alertReasons.allSatisfy({ ["offset_lag", "quarantined_record"].contains($0) }) == true,
                  states(c, "ingest", [.red, .amber]).allSatisfy({
                      $0.blind != true && ($0.blindCodes ?? []).isEmpty
                          && ($0.contextCodes ?? []).allSatisfy { ["offset_lag", "quarantined_record"].contains($0) }
                  }) else { return false }
            if c.watcher?.catchUp != nil { return producerCatchingUp(c) }
            return lagShown(c) && c.watcher?.alertReasons.allSatisfy { ["offset_lag", "quarantined_record"].contains($0) } == true
        }
        /// A2: the producer’s estimate as `{span}` words, only while it is catching up.
        static func lagEta(_ c: Capture) -> String? { producerCatchingUp(c) ? c.watcher?.catchUp?.etaS.flatMap(span) : nil }
        /// A4: `ingest` red, and the watcher file readable but no longer fresh.
        static func readerStopped(_ c: Capture) -> Bool { red(c, "ingest") && c.watcher.map { $0.readable && !$0.fresh } == true }
        /// The words for the bytes left to read and the episode’s length, or nil when the fact is not in the capture.
        static func lagSize(_ c: Capture) -> String? { (c.watcher?.catchUp?.bytesBehind ?? c.watcher?.lagBytes).flatMap(size) }
        static func lagSpan(_ c: Capture) -> String? { c.lagEpisodeStart.flatMap { span(c.now.timeIntervalSince($0)) } }

        // MARK: the calm form (A2)

        static let allGood = S("S1", .reassurance, "All good") { c, _, _ in allGoodProven(c) }
        /// P2c (R2-B2): the headline is not proof on its own; All good itself is proven, as the oracle proves it.
        static let backedUp = S("S01", .claim, "Backed up to Google Drive {at}.") { c, p, v in
            allGoodProven(c) && p.headline == allGood.text && allGoodDate(c).map { at($0, c) } == v[0]
        }
        static let checkSoon = S("R31", .claim, "Check soon") { c, p, _ in
            p.count == 0 && appEscalationCauses(c).isEmpty && (c.verdict?.checks.contains { $0.state.uiKind == .notice } == true
                || c.watcher.map { w in w.readable && w.fresh && w.alertReasons.contains { !["offset_lag", "quarantined_record"].contains($0) && !($0 == "coverage_drop" && w.dbProbeFailed) } } == true)
        }
        static let checkingMemory = S("X01", .neutral, "Checking on your memory…")
        static let checkingSleep = S("X02", .claim, "Checking after sleep…") { c, _, _ in c.wakeCatchUpPending }
        static let notConfirmed = S("X01.detail", .neutral, "Saving, search, backups and upkeep aren’t confirmed yet.")
        static let lastBackup = S("S4", .claim, "Last backup {at}.") { c, _, v in c.lastKnownBackupAt.map { $0 <= c.now && sayable($0) && at($0, c) == v[0] } == true }
        static let r02Headline = S("R02.headline", .claim, "Catching up on new conversations", .remembering) { c, _, _ in catchingUp(c) }
        static let r02 = S("R02", .claim, "Catching up on new conversations.", .remembering) { c, _, _ in catchingUp(c) }
        static let s03Headline = S("S03.headline", .claim, "Backing up now") { c, _, _ in runningUpload(c) != nil }
        static let s03 = S("S03", .claim, "To Google Drive.", .safe) { c, _, _ in runningUpload(c) != nil }
        static let s03Started = S("S03.started", .claim, "Started at {HH:mm}.") { c, _, v in started(c, today: true, v[0]) }
        static let s03StartedOn = S("S03.startedOn", .claim, "Started {at}.") { c, _, v in started(c, today: false, v[0]) }
        static let s03Progress = S("S03.progress", .claim, "{p}% done.") { c, _, v in progress(c, v[0]) }
        static let s03ProgressEta = S("S03.progressEta", .claim, "{p}% done, about {HH:mm}.") { c, _, v in
            progress(c, v[0]) && runningUpload(c)?.onSchedule?.eta.map { calendar(c).isDate(minute($0, c), inSameDayAs: c.now) && hhmm($0, c) == v[1] } == true
        }
        static let catchingTwo = S("C3", .claim, "Catching up on {two}") { c, _, v in
            v[0].components(separatedBy: " and ").allSatisfy { behind($0, c) }
        }
        /// Three things or more Behind; only two Behind rows exist before P4 (R02, S03), so it cannot hold yet.
        static let catchingFew = S("C.few", .claim, "Catching up on a few things") { c, _, _ in [catchingUp(c), runningUpload(c) != nil].filter { $0 }.count >= 3 }
        static let f02 = S("F02", .neutral, "Can’t confirm agents can search right now.", .finding)
        static let checking = S("R17.checking", .neutral, "Checking…")
        static let cantConfirmBackups = S("C.backups", .neutral, "Can’t confirm backups yet")
        static let stillChecking = S("C.one", .neutral, "Still checking {thing}")
        static let stillCheckingFew = S("C.few.checking", .neutral, "Still checking a few things")
        static let fineIs = S("A2.fine.is", .reassurance, "{fine} is fine.") { c, p, v in reassuranceCoverage("A2.fine.is", v, p, c).map { $0.isSubset(of: provenFine(c)) } == true }
        static let fineAre = S("A2.fine.are", .reassurance, "{fines} are fine.") { c, p, v in reassuranceCoverage("A2.fine.are", v, p, c).map { $0.isSubset(of: provenFine(c)) } == true }
        static let everythingElse = S("M5", .reassurance, "Everything else is fine.") { c, p, _ in reassuranceCoverage("M5", [], p, c).map { $0.isSubset(of: provenFine(c)) } == true }

        // MARK: the needs form (A1)

        static let countOne = S("A1.count.one", .claim, "1 thing needs you") { _, p, _ in countIs(p, "1") }
        static let countMany = S("A1.count.many", .claim, "{N2} things need you") { _, p, v in countIs(p, v[0]) }
        /// R1.11 (L52, L53), D5: the safety net. It says what it knows and what it doesn’t; it should never show, and stays
        /// true if it does (the report carries the check names). It replaces R17’s generic sentence everywhere.
        static let d5 = S("D5", .neutral, "A check failed, and BrainBar can’t tell which")
        static let d5Line = S("D5.line", .neutral, "Open BrainBar for the details.")
        /// R1.11: one sentence that says who acts, only on a row where an agent really must (replaces U12 and U12.give).
        static let agent = S("R111.agent", .neutral, "An agent needs to look at this.")
        /// C5: the running upload’s row (replaces D10).
        static let ifNotFinished = S("C5", .neutral, "If it doesn’t finish, an agent needs to look at it.")

        // MARK: R1.11 (CHANGES-R1.11 § 1 and § 6, approved by L53): each row names the thing

        /// A1, A2: the reader is behind past its limit (the app’s clock) or `ingest` is red, on a fresh watcher file with lag.
        static let lagBehind = S("A1", .claim, "Reading new conversations is behind", .remembering) { c, _, _ in lagShown(c) }
        static let lagLeftFor = S("A1.leftFor", .claim, "{size} left to read, behind for {span}.") { c, _, v in
            lagShown(c) && lagSize(c) == v[0] && lagSpan(c) == v[1]
        }
        static let lagLeft = S("A1.left", .claim, "{size} left to read.") { c, _, v in lagShown(c) && lagSize(c) == v[0] }
        static let lagFor = S("A1.for", .claim, "Behind for {span}.") { c, _, v in lagShown(c) && lagSpan(c) == v[0] }
        static let byItself = S("A1.byItself", .claim, "It’s catching up by itself.") { c, _, _ in lagDraining(c) }
        /// A2 (approved words #3): with the producer’s estimate.
        static let byItselfEta = S("A2.byItself", .claim, "It’s catching up by itself, about {span} to go.") { c, _, v in
            lagDraining(c) && lagEta(c) == v[0]
        }
        /// A3 (approved words #4, its title): the producer says the lag stopped shrinking.
        static let notCatchingUp = S("A3", .claim, "Reading new conversations isn’t catching up", .remembering) { c, _, _ in producerStalled(c) }
        /// A4: `ingest` red and the watcher file readable but not fresh.
        static let stopped = S("A4", .claim, "Reading new conversations has stopped", .remembering) { c, _, _ in readerStopped(c) }
        static let lastRead = S("A4.lastRead", .claim, "It last read {at}.") { c, _, v in
            readerStopped(c) && c.watcher?.lastReadAt.map { $0 <= c.now && sayable($0) && at($0, c) == v[0] } == true
        }
        static let coverage = S("A5", .claim, "Some new conversations aren’t being read", .remembering) { c, _, _ in coverageDrop(c) }
        static let coverageLine = S("A5.line", .claim, "Fewer of them are turning into memories than usual.") { c, _, _ in coverageDrop(c) }
        static let unreadFiles = S("A6", .claim, "Some conversation files can’t be read", .remembering) { c, _, _ in
            fresh(c, "file_ingestion_failure")
        }
        static let unreadFilesLine = S("A6.line", .claim, "BrainBar skips them and reads the rest.") { c, _, _ in fresh(c, "file_ingestion_failure") }
        static let providerUnread = S("A7", .claim, "{provider}’s sessions aren’t being read", .remembering) { c, _, v in providers(c).contains(v[0]) }
        static let providerUnreadLine = S("A7.line", .claim, "Its new work isn’t reaching memory.") { c, _, _ in !providers(c).isEmpty }
        static let savesStuck = S("B1", .claim, "Saves are stuck", .remembering) { c, _, _ in red(c, "stores") }
        static let savesStuckLine = S("B1.line", .claim, "The oldest has waited {span} and isn’t in memory yet.") { c, _, v in
            states(c, "stores", [.red]).contains { $0.ageS.flatMap(span) == v[0] }
        }
        static let savesSlow = S("B2", .claim, "Saves are slow", .remembering) { c, _, _ in amber(c, "stores") }
        static let savesSlowLine = S("B2.line", .claim, "The oldest has waited {span}.") { c, _, v in
            states(c, "stores", [.amber]).contains { $0.ageS.flatMap(span) == v[0] }
        }
        static let searchDown = S("B3", .claim, "Agents can’t search memory", .finding) { c, _, _ in red(c, "search") }
        static let searchFails = S("B3.line", .claim, "Search has failed its last {N2} checks.") { c, _, v in
            states(c, "search", [.red]).contains { $0.fails.map { $0 >= 2 && String($0) == v[0] } == true }
        }
        static let searchCheck = S("B4", .claim, "Search failed a check", .finding) { c, _, _ in amber(c, "search") }
        static let searchOneOff = S("B4.line", .neutral, "It may be a one-off; BrainBar checks again every 5 min.")
        static let localLate = S("C1.local", .claim, "The local copy is late", .safe) { c, _, _ in red(c, "backup_local") }
        static let driveLate = S("C1.drive", .claim, "The Google Drive copy is late", .safe) { c, _, _ in red(c, "backup_offsite") && driveSignedIn(c) }
        static let conversationsLate = S("C1.conversations", .claim, "The conversations copy in Google Drive is late", .safe) { c, _, _ in
            red(c, "transcripts") && driveSignedIn(c)
        }
        static let newestIs = S("C1.newest", .claim, "The newest is {age}.") { c, _, v in
            ["backup_local", "backup_offsite", "transcripts"].contains { age(c, [$0], v[0]) }
        }
        static let localRunningLate = S("C4.local", .claim, "The local copy is running late", .safe) { c, _, _ in amber(c, "backup_local") }
        static let driveRunningLate = S("C4.drive", .claim, "The Google Drive copy is running late", .safe) { c, _, _ in amber(c, "backup_offsite") }
        static let conversationsRunningLate = S("C4.conversations", .claim, "The conversations copy in Google Drive is running late", .safe) { c, _, _ in
            amber(c, "transcripts")
        }
        static let noRoom = S("C7", .claim, "Not enough room for the next backup", .safe) { c, _, _ in red(c, "disk") }
        static let roomTight = S("C8", .claim, "Room for backups is getting tight", .safe) { c, _, _ in amber(c, "disk") }
        static let checksStopped = S("D2", .claim, "BrainBar’s checks have stopped") { c, _, _ in monitorLate(c) }
        static let checksLastRan = S("D2.line", .claim, "The last one ran {at}, so nothing since then is confirmed.") { c, _, v in
            monitorLate(c) && c.verdict.map { $0.evaluatedAt <= c.now && sayable($0.evaluatedAt) && at($0.evaluatedAt, c) == v[0] } == true
        }
        static let unknownCheck = S("D3", .claim, "A check this BrainBar doesn’t know is failing") { c, _, _ in
            c.verdict?.checks.contains { !known.contains($0.name) && [.red, .amber].contains($0.state) } == true
        }
        static let newerLayer = S("D3.newer", .neutral, "BrainLayer is newer than BrainBar.")
        static let updateNames = S("D3.update", .neutral, "Updating BrainBar will name it.")
        /// D4: a row from a verdict that is no longer fresh says when it was true.
        static let asOf = S("D4", .claim, "(as of {at})") { c, _, v in
            !verdictIsFresh(c) && c.verdict.map { $0.evaluatedAt <= c.now && sayable($0.evaluatedAt) && at($0.evaluatedAt, c) == v[0] } == true
        }

        static let u03 = S("U03", .claim, "Nightly check-up failed {N3} nights", .upkeep) { c, _, v in  // R1.12 (I 6)
            appEscalationCauses(c).contains(appNotificationCause("maintenance"))
                && c.maintenanceNightFailures.map { String($0) == v[0] } == true
        }
        static let stillSaved = S("U03.stillSaved", .reassurance, "Your memory is still being saved and backed up, as of the last checks.") { c, p, v in
            reassuranceCoverage("U03.stillSaved", v, p, c).map { $0.isSubset(of: provenFine(c)) } == true
        }
        static let s19 = S("S19", .claim, "The newest checked backup is damaged", .safe) { c, _, _ in red(c, "integrity") }
        static let s19Memory = S("S19.memory", .claim, "Your memory itself may be damaged too.") { c, _, _ in red(c, "integrity") }
        static let f07 = S("F07", .claim, "Search by meaning isn’t catching up", .finding) { c, _, _ in stalled(c) }
        static let f07Since = S("F07.since", .claim, "Its queue hasn’t shrunk{over}.") { c, _, v in
            stalled(c) { ($0.ageS.flatMap(over).map { " " + $0 } ?? "") == v[0] }  // one exact form: `{over}` of a stalled check's age
        }
        static let exactWords = S("F07.exactWords", .reassurance, "Agents can still find every memory by exact words.") { c, _, _ in
            c.keywordCoverage == 1 && searchProven(c)
        }
        /// R21, R25: no copy of any kind on record, the local check red with no clock and no Drive clock of either kind.
        static let noRecord = S("S07", .claim, "No verified backup on record", .safe) { c, _, _ in
            let local = c.verdict?.checks.filter { $0.name == "backup_local" } ?? []
            return noDriveClock(c) && !local.isEmpty && local.allSatisfy { $0.state == .red && $0.ageS == nil }
        }
        static let nothingInDrive = S("R14", .claim, "Nothing is backed up to Google Drive yet.") { c, _, _ in noDriveClock(c) && c.backupsMeasured }
        static let s32 = S("S32", .claim, "A backup to Google Drive started at {HH:mm} and is still running.", .safe) { c, _, v in
            started(c, today: true, v[0])
        }
        static let s32Progress = S("S32.progress", .claim, "A backup to Google Drive started at {HH:mm} and is {p}% done.", .safe) { c, _, v in
            started(c, today: true, v[0]) && progress(c, v[1])
        }
        static let s32On = S("S32.on", .claim, "A backup to Google Drive started {at} and is still running.", .safe) { c, _, v in
            started(c, today: false, v[0])
        }
        static let s32ProgressOn = S("S32.progressOn", .claim, "A backup to Google Drive started {at} and is {p}% done.", .safe) { c, _, v in
            started(c, today: false, v[0]) && progress(c, v[1])
        }
        static let s06 = S("S06", .claim, "The last {N2} backups didn’t finish", .safe) { c, _, v in
            c.backupDailyFailures.map { $0 >= 2 && String($0) == v[0] } == true
        }
        static let s06Newest = S("S06.newest", .claim, "Your newest backup is {age}.") { c, _, v in
            c.backupsMeasured && age(c, ["backup_local", "backup_offsite"], v[0])
        }
        static let s06Confirmed = S("S06.confirmed", .claim, "The newest backup BrainBar can confirm is {age}.") { c, _, v in
            age(c, ["backup_local", "backup_offsite"], v[0])
        }
        static let r15 = S("R15", .claim, "Your newest backup in Google Drive is {age}.", .safe) { c, _, v in c.backupsMeasured && age(c, ["backup_offsite"], v[0]) }
        static let s13 = S("S13", .claim, "Google Drive is signed out", .safe) { c, _, _ in drive(c, missing: false) }
        static let s13Missing = S("S13.missing", .claim, "Google Drive isn’t connected", .safe) { c, _, _ in drive(c, missing: true) }
        static let s13Detail = S("S13.detail", .claim, "Backups can’t upload until you reconnect it.") { c, _, _ in drive(c, missing: false) }
        static let s13DetailMissing = S("S13.detailMissing", .claim, "Backups can’t upload until you connect it.") { c, _, _ in drive(c, missing: true) }
        static let d13 = S("D13", .claim, "Conversation backups can’t upload either until Drive is reconnected.", .safe) { c, _, _ in drive(c, missing: false) }
        /// R26: under the never-connected row.
        static let d13Missing = S("D13.missing", .claim, "Conversation backups can’t upload either until Drive is connected.", .safe) { c, _, _ in
            drive(c, missing: true)
        }

        // MARK: action labels and the standing items (R4)

        static let reconnect = S("action.reconnect", .neutral, "Reconnect Google Drive")
        static let connect = S("action.connect", .neutral, "Connect Google Drive")
        static let copyReport = S("action.copyReport", .neutral, "Copy a report for your agent")
        static let storage = S("action.storage", .neutral, "Open Storage Settings")
        static let settings = S("R4.settings", .neutral, "Settings…")
        static let quit = S("R4.quit", .neutral, "Quit BrainBar")
        static let restart = S("R4.restart", .neutral, "Restart BrainBar")
        static let settingsKey = "⌘,", quitKey = "⌘Q"

        static let all: [S] = [
            allGood, backedUp, checkingMemory, checkingSleep, notConfirmed, lastBackup, r02Headline, r02, s03Headline, s03, s03Started, s03StartedOn,
            s03Progress, s03ProgressEta, catchingTwo, catchingFew, f02, checking, cantConfirmBackups, stillChecking, stillCheckingFew,
            fineIs, fineAre, everythingElse, countOne, countMany, d5, d5Line, agent, ifNotFinished, u03, stillSaved, s19,
            checkSoon, s19Memory, f07, f07Since, exactWords, noRecord, nothingInDrive, s32, s32Progress, s32On, s32ProgressOn, s06, s06Newest, s06Confirmed, r15, s13,
            s13Missing, s13Detail, s13DetailMissing, d13, d13Missing, reconnect, connect, copyReport, storage, settings, quit, restart,
            lagBehind, lagLeftFor, lagLeft, lagFor, byItself, byItselfEta, notCatchingUp, stopped, lastRead, coverage, coverageLine, unreadFiles, unreadFilesLine,
            providerUnread, providerUnreadLine, savesStuck, savesStuckLine, savesSlow, savesSlowLine, searchDown, searchFails, searchCheck,
            searchOneOff, localLate, driveLate, conversationsLate, newestIs, localRunningLate, driveRunningLate, conversationsRunningLate,
            noRoom, roomTight, checksStopped, checksLastRan, unknownCheck, newerLayer, updateNames, asOf,
        ]
        static let exact = Dictionary(uniqueKeysWithValues: all.filter { !$0.template.contains("{") }.map { ($0.template, $0) })
        static let templated = all.filter { $0.template.contains("{") }

        // MARK: the words a placeholder can take

        /// A thing's noun in a Behind or Can't-tell headline, and in a "… fine." line.
        static let behindNoun: [Thing: String] = [.remembering: "new conversations", .finding: "search", .safe: "backups", .upkeep: "upkeep"]
        static let fineNoun: [Thing: String] = [.remembering: "saving", .finding: "search", .safe: "backups", .upkeep: "upkeep"]

        /// "saving, search and upkeep": the words joined; `list` capitalises the first.
        static func joined(_ words: [String]) -> String { words.count < 2 ? words.joined() : words.dropLast().joined(separator: ", ") + " and " + words.last! }
        static func list(_ words: [String]) -> String { let s = joined(words); return s.prefix(1).uppercased() + s.dropFirst() }

        /// The things a list of nouns names, distinct and in the things' order, read leniently (any case, any order, repeats).
        static func things(_ value: String, _ nouns: [Thing: String]) -> [Thing] {
            let words = Set(value.lowercased().replacingOccurrences(of: " and ", with: ", ").components(separatedBy: ", "))
            return Thing.allCases.filter { nouns[$0].map(words.contains) == true }
        }

        /// The things a "… fine." list names.
        static func fineThings(_ value: String) -> Set<Thing> { Set(things(value, fineNoun)) }

        /// Whether `value` is a word its slot's formatter makes for this capture (P2c, round 2 of #1788, R2-B1). A round trip:
        /// the value is read back, leniently, to the quantity it names (a moment in the capture's zone, a duration, a count, a
        /// list of things), the presenter's own formatter says that quantity again, and only a byte-for-byte equal result is
        /// accepted, so a lenient reading can never widen what matches. One part is not a round trip: the floors of N2 and N3.
        /// A count is said by `String(_:)`, but its floor (2, 3) is the range of its template, which the presenter picks by
        /// the count, not a property of any formatter; it stays an explicit range, with negative tests.
        static func fits(_ slot: String, _ value: String, _ c: Capture) -> Bool {
            switch slot {
            case "N2", "N3": return Int(value).map { $0 >= (slot == "N2" ? 2 : 3) && String($0) == value } == true
            case "p": return Double(value).map { (0...1).contains($0 / 100) && String(percent($0 / 100)) == value } == true
            // A moment's words: first the formatter's own words for a moment the capture holds (`spoken`), which the presenter's
            // words match by construction (round 1 of #1790); then a reading back, for values not the capture's own.
            case "HH:mm": return spoken(c).contains { hhmm($0, c) == value } || clock(value).flatMap { on(c.now, $0, c) }.map { hhmm($0, c) == value } == true
            case "at": return spoken(c).contains { at($0, c) == value } || moments(value, c).contains { at($0, c) == value }
            case "age":
                let hours = value.split(separator: " ").first.flatMap { Double($0) }.map { [c.now.addingTimeInterval(-$0 * 3600)] } ?? []
                return (spoken(c) + hours + (value.hasPrefix("from ") ? moments(String(value.dropFirst(5)), c) : [])).contains { copyAge(since: $0, c) == value }
            case "over":
                // A stall's age: just over the count of units the words name. The empty form is the formatter's for no age over a minute.
                let words = value.split(separator: " "), unit = words.last.map { $0.hasPrefix("hour") ? 3600.0 : $0.hasPrefix("minute") ? 60 : 0 } ?? 0
                let n = words.count < 2 ? nil : ["a", "an"].contains(words[words.count - 2]) ? 1 : Double(words[words.count - 2])
                return value.isEmpty || n.map { over($0 * unit + 1).map { " " + $0 } == value } == true
            // R1.11: the words its formatter makes for a quantity the capture holds, byte for byte.
            // Read back to the quantity the words name, said again, byte for byte; whether it is the capture's is the condition's.
            case "size":
                let parts = value.split(separator: "\u{00A0}")
                guard parts.count == 2, let n = Double(parts[0]), n.isFinite, n > 0, n < 1e9 else { return false }
                let bytes = Int((n * (parts[1] == "GB" ? 1_073_741_824 : parts[1] == "MB" ? 1_048_576 : .nan)).rounded(.down))
                return size(bytes) == value
            case "span":
                let parts = value.split(separator: "\u{00A0}")
                guard parts.count == 2, let n = Int(parts[0]), n > 0, n < 1_000_000,
                      let unit = ["min": 60.0, "h": 3600, "d": 86_400][String(parts[1])] else { return false }
                return span(Double(n) * unit) == value
            case "provider": return providerNames.values.contains(value)  // L38's names; whether one is named is the condition's
            case "thing", "two", "fine", "fines":
                let nouns = slot.hasPrefix("fine") ? fineNoun : behindNoun, named = things(value, nouns)
                let said = (slot.hasPrefix("fine") ? list : joined)(named.compactMap { nouns[$0] })
                return said == value && (slot == "two" ? named.count == 2 : slot == "fines" ? named.count >= 2 : named.count == 1)
            default: return false
            }
        }

        /// Every moment the presenter says as `{at}`, `{HH:mm}` or `{age}`: the last known backup, the all-good copy, the running
        /// upload's start and estimate, and every copy a database backup check dates. The property test fails if the derivation
        /// ever says a moment that is not here.
        static func spoken(_ c: Capture) -> [Date] {
            // Each as the derivation may say it (the last backup at or before now, the estimate only today, a copy `copyAge`
            // dates), and only a moment the formatter has words for (CI on #1790).
            let last = c.lastKnownBackupAt.flatMap { $0 <= c.now && sayable($0) ? $0 : nil }
            let eta = runningUpload(c)?.onSchedule?.eta.flatMap { calendar(c).isDate(minute($0, c), inSameDayAs: c.now) ? $0 : nil }
            let copies = (c.verdict?.checks ?? []).filter { ["backup_local", "backup_offsite"].contains($0.name) && valid($0.ageS) }
                .map { c.verdict!.evaluatedAt.addingTimeInterval(-$0.ageS!) }.filter { copyAge(since: $0, c) != nil }
            // R1.11: the watcher's last read (A4) and the verdict's own time (D2, D4), each at or before now.
            let read = c.watcher?.lastReadAt.flatMap { $0 <= c.now && sayable($0) ? $0 : nil }
            let evaluated = c.verdict.flatMap { $0.evaluatedAt <= c.now && sayable($0.evaluatedAt) ? $0.evaluatedAt : nil }
            return ([last, allGoodDate(c), uploadStart(c)?.date, eta, read, evaluated].compactMap { $0 } + copies).filter { hasWords($0, c) }
        }

        /// "HH:mm" read leniently as an hour and a minute.
        private static func clock(_ value: String) -> (hour: Int, minute: Int)? {
            let parts = value.split(separator: ":").map { Int($0) }
            guard parts.count == 2, let hour = parts[0], let minute = parts[1] else { return nil }
            return (hour, minute)
        }

        /// The moment at that clock time on `day`'s date in the capture's zone (a time the day skips moves, and so says
        /// another clock time).
        private static func on(_ day: Date, _ time: (hour: Int, minute: Int), _ c: Capture) -> Date? {
            sayable(day) ? calendar(c).date(bySettingHour: time.hour, minute: time.minute, second: 0, of: day) : nil
        }

        /// The moments `{at}` words can name, all at or before now (every `{at}` is a past event): "today" or "yesterday" at a
        /// clock time, or a month and day at one in any of the last nine years (no year is said; Feb 29 needs a leap year).
        /// Words name a minute and no year, so reading them back is lossy at the edges (a repeated minute's second pass, a moment
        /// just inside a day, one over nine years old): only values that are not the capture's own depend on it.
        static func moments(_ value: String, _ c: Capture) -> [Date] {
            let halves = value.components(separatedBy: " at "), cal = calendar(c)
            guard halves.count == 2, let time = clock(halves[1]), sayable(c.now) else { return [] }
            let day = halves[0].split(separator: " ").map(String.init)
            var dates: [Date?] = []
            if day == ["today"] || day == ["yesterday"] {
                dates = [cal.date(byAdding: .day, value: day == ["today"] ? 0 : -1, to: c.now).flatMap { on($0, time, c) }]
            } else if day.count == 2, let month = formatter("MMM", c).shortMonthSymbols.firstIndex(of: day[0]), let d = Int(day[1]) {
                let year = cal.component(.year, from: c.now)
                dates = (0...8).map { cal.date(from: DateComponents(year: year - $0, month: month + 1, day: d, hour: time.hour, minute: time.minute)) }
            }
            return dates.compactMap { $0 }.filter { $0 <= c.now && sayable($0) }
        }
    }

    // MARK: words for times (truth table Part 0): Gregorian, in the capture's time zone

    /// One formatter per format and zone on each thread, reused: making one per call was most of the enumeration's time, and
    /// a per-thread cache needs no lock.
    static func formatter(_ format: String, _ c: Capture) -> DateFormatter {
        let key = "BrainBarMenuPresenter.formatter|" + format + "|" + c.timeZone.identifier
        if let made = Thread.current.threadDictionary[key] as? DateFormatter { return made }
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.calendar = Calendar(identifier: .gregorian)
        f.timeZone = c.timeZone
        f.dateFormat = format
        Thread.current.threadDictionary[key] = f
        return f
    }

    /// Drop fractional seconds before day/clock words; whole seconds preserve historical offsets with leftover seconds.
    static func minute(_ date: Date, _ c: Capture) -> Date {
        guard sayable(date) else { return date }
        return Date(timeIntervalSinceReferenceDate: date.timeIntervalSinceReferenceDate.rounded(.down))
    }

    static func hhmm(_ date: Date, _ c: Capture) -> String { formatter("HH:mm", c).string(from: minute(date, c)) }

    /// One Gregorian calendar per zone on each thread, reused like the formatters (P2c: the matcher's round trip reads with it).
    static func calendar(_ c: Capture) -> Calendar {
        final class Made { let calendar: Calendar; init(_ calendar: Calendar) { self.calendar = calendar } }
        let key = "BrainBarMenuPresenter.calendar|" + c.timeZone.identifier
        if let made = Thread.current.threadDictionary[key] as? Made { return made.calendar }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = c.timeZone
        Thread.current.threadDictionary[key] = Made(calendar)
        return calendar
    }

    /// `{at}`, a past event: "today at 04:00", "yesterday at 23:10" (both under 24 h), else "Nov 9 at 03:44".
    static func at(_ date: Date, _ c: Capture) -> String {
        let calendar = calendar(c), spoken = minute(date, c)
        if c.now.timeIntervalSince(date) < 86_400 {
            if calendar.isDate(spoken, inSameDayAs: c.now) { return "today at \(hhmm(spoken, c))" }
            if let yesterday = calendar.date(byAdding: .day, value: -1, to: c.now), calendar.isDate(spoken, inSameDayAs: yesterday) {
                return "yesterday at \(hhmm(spoken, c))"
            }
        }
        return formatter("MMM d 'at' HH:mm", c).string(from: minute(date, c))
    }

    /// A time the words can name: Foundation formats a time outside `distantPast…distantFuture` (or not finite) as a
    /// date that does not exist ("Sep 20 at 02:00" for −∞), so no sentence names one (P2b, found by writing S4's condition).
    static func sayable(_ date: Date) -> Bool { (Date.distantPast...Date.distantFuture).contains(date) }

    /// The formatter has words for this moment. An older Foundation (macOS 15, CI on #1790) formats some moments in
    /// `sayable`'s range as "", so a moment with no words is one the presenter cannot say.
    static func hasWords(_ date: Date, _ c: Capture) -> Bool { !hhmm(date, c).isEmpty && !formatter("MMM d 'at' HH:mm", c).string(from: minute(date, c)).isEmpty }

    static func percent(_ progress: Double) -> Int { Int((progress * 100).rounded()) }

    static func valid(_ age: Double?) -> Bool { age.map { $0.isFinite && $0 >= 0 } == true }

    /// `{copy age}` at `now`: the copy's time is `evaluatedAt − ageS` (wall-clock for backups, U10), and its age is counted
    /// from that time to `now`, never from `evaluatedAt` (a verdict can stay awake-fresh across hours of sleep): "21 hours
    /// old" under a day, else "from Nov 9 at 03:44". Nil for no, a negative or non-finite age, or a time after now.
    static func copyAge(_ age: Double?, _ c: Capture, _ v: BrainBarHealthVerdict) -> String? {
        guard let age, valid(age) else { return nil }
        return copyAge(since: v.evaluatedAt.addingTimeInterval(-age), c)
    }

    /// `{copy age}` of a copy made at `copied`, at `now`.
    static func copyAge(since copied: Date, _ c: Capture) -> String? {
        let elapsed = c.now.timeIntervalSince(copied)
        guard sayable(copied), elapsed.isFinite, elapsed >= 0 else { return nil }
        guard elapsed >= 86_400 else {
            let hours = min(23, max(1, Int((elapsed / 3600).rounded())))
            return hours == 1 ? "1 hour old" : "\(hours) hours old"
        }
        return hasWords(copied, c) ? "from \(at(copied, c))" : nil
    }

    /// R1.11 `{span}`: how long, rounded down ("40 min", "2 h", "3 d"); nil for a negative or non-finite one.
    static func span(_ seconds: Double) -> String? {
        guard seconds.isFinite, seconds >= 0, seconds < 1e12 else { return nil }
        if seconds < 3600 { return "\(max(1, Int(seconds / 60)))\u{00A0}min" }
        if seconds < 172_800 { return "\(Int(seconds / 3600))\u{00A0}h" }
        return "\(Int(seconds / 86_400))\u{00A0}d"
    }

    /// R1.11 `{size}`: bytes left to read, in the Dashboard's binary megabytes ("117 MB", "1.4 GB"); nil under one megabyte.
    static func size(_ bytes: Int) -> String? {
        let mb = Double(bytes) / 1_048_576
        guard mb >= 1 else { return nil }
        if mb < 1024 { return "\(Int(mb.rounded()))\u{00A0}MB" }
        return String(format: "%.1f\u{00A0}GB", mb / 1024)
    }

    /// `{over d}`: a strict lower bound from an awake-adjusted age, rounded down ("in over an hour"); nil under a minute.
    static func over(_ age: Double) -> String? {
        guard age.isFinite, age > 60 else { return nil }
        func below(_ unit: Double) -> Int { Int((min(age, 1e12) / unit).rounded(.up)) - 1 }
        if below(3600) >= 2 { return "in over \(below(3600)) hours" }
        if below(3600) == 1 { return "in over an hour" }
        return below(60) == 1 ? "in over a minute" : "in over \(below(60)) minutes"
    }
}

// Notification-only uncertainty labels live with the wording registry, outside the menu sentence inventory.
extension BrainBarMenuPresenter {
    static func genericNotificationTitle(_ thing: Thing) -> String {
        switch thing {
        case .remembering: "Conversation saving needs attention"
        case .finding: "Memory search needs attention"
        case .safe: "Backups need attention"
        case .upkeep: "Upkeep needs attention"
        case .system: "BrainLayer needs attention"
        }
    }
}

// The new design's surface sentences live with the wording registry, outside the menu sentence inventory (its own
// labels and state words are the spec's copy deck). `BrainBarNewDesignWordsTests` keeps sentences out of NewDesign/.
extension BrainBarMenuPresenter {
    enum SurfaceSay {
        static let diagnostics = "Diagnostics"
        static let openDiagnostics = "Open"
        static func diagnosticState(_ state: BrainBarHealthVerdict.State) -> String {
            switch state { case .green: "OK"; case .amber: "Warning"; case .red: "Problem"; case .unknown: "Not read" }
        }
        static let copyFullReport = "Copy a full report for your agent"
        static let checksUnreadable = "Checks have not been read yet."
        static let settingsBackupsUnreadable = "Settings backups could not be counted."
        static func settingsBackups(_ count: Int, path: String) -> String {
            "\(count) settings \(count == 1 ? "backup" : "backups") in \(path)."
        }
        static let meaningModel = "Search by meaning"
        static func modelExplanation(_ name: String, namesModel: Bool = true) -> [String] {
            ["\(namesModel ? name : "This model") turns memories into searchable meaning on this Mac.",
             "Search by meaning uses it. Keyword search does not need it."]
        }
        static let backupUploading = "Uploading to Google Drive"
        static let backupVerifying = ObservabilityPresentation.backupVerificationText
        /// The Backups page: the newest verified copy, dated by when it was made (`db_snapshot.last_at` is the snapshot's
        /// creation time, not the time it was verified, so it never reads as "verified at").
        static let backupLastGood = "Last verified copy made"
        static let backupUploadStarted = "Upload started"
        /// The time of the last confirmed byte checkpoint, beside the percentage: a clock time, never a duration.
        static func backupUpdatedAt(_ time: String) -> String { "updated \(time)" }
        /// Google Drive's status line in the Pipeline while a backup uploads (the station's own line, 11 pt).
        static let backupUploadingShort = "Uploading"
        static let backupVerifyingShort = "Checking copy"
        static func driveUploadingPercent(_ percent: Int) -> String { "Uploading \(percent)%" }
        static func driveStalledSince(_ time: String) -> String { "Stalled since \(time)" }
        static let backupInProgress = "In progress"
        static func driveCopyUploading(_ age: String) -> String {
            "Your database backup in Google Drive is \(age); an upload is running."
        }
        /// R3: a reconnect in progress.
        static let waitingForBrowser = "Waiting for you to approve in your browser…"
        /// R12: a failed copy leaves the button as it was and says so under it.
        static let copyFailed = "Couldn’t copy the report. Try again."
        /// Backups page, Google Drive connected (comp 09).
        static let driveUploadsHere = "Database and conversation backups upload here."
        /// Design pre-pass 5b: why an area with known values still carries the unknown ring.
        static let unconfirmed = "not confirmed by the last check"
        /// Review fix 7 (lead's copy ruling L7, approved): a switched-off job's number cell, and its switch's help tags.
        static let off = "off"
        static func turnOff(_ job: String) -> String { "Turn off \(job)" }
        static func turnOn(_ job: String) -> String { "Turn on \(job)" }
        /// Review polish 5 (lead's copy ruling L7, approved): the Jobs switch column's header, and an always-running job's
        /// state in place of "always on".
        static let onHeader = "On"
        static let running = "running"
        /// Lead's ruling L5/L7 (approved): the Agents column's caption. It counts every memory attributed to an agent family
        /// in 24 h (observability emitters), the same noun as the title bar's count, not saves.
        static let memoriesIn24h = "memories in 24 h"

        // MARK: R1 copy deck (DESIGN-SPEC-R1 § 4; approved by L16). Every word the redrawn surfaces draw.

        // Dropdown
        static let activity = "Activity"
        static let search = "Search"
        static let backups = "Backups"
        static let jobs = "Jobs"
        static let fine = "fine"
        static let failed = "failed"
        static let late = "late"
        static let openBrainBar = "Open BrainBar"
        static func ready(_ percent: Int) -> String { "\(percent)\u{00A0}% ready" }
        static func ago(_ age: String) -> String { "\(age) ago" }
        static func lastHour(saved: String, read: String) -> String { "Last hour: \(saved) saved by agents, \(read) read" }
        static func searchable(_ percent: String) -> String { "\(percent)\u{00A0}% of memories are searchable by meaning" }
        static func thingsNeedYou(_ n: Int) -> String { "\(n) things need you" }

        // Window
        static let overview = "Overview"
        static func checkedAgo(_ age: String) -> String { "Checked \(age) ago" }
        static func nextCheckIn(_ age: String) -> String { "Next check in \(age)" }
        static func lastCheckedAgo(_ age: String) -> String { "Last checked \(age) ago" }
        static let savedByAgents = "saved by agents"
        static let readFromConversations = "session lines ingested"
        static let readsByAgents = "reads by agents"
        static func agentReadsAccessibility(_ count: Int?) -> String {
            guard let count else { return "Reads by agents unavailable." }
            return "\(count) reads by agents."
        }
        static let saved = "saved"
        static let sessionLinesWord = "lines"
        static let now = "now"
        static let pipeline = "Pipeline"
        static let goesIn = "Goes in"
        static let keptSafe = "Kept safe"
        static let comesOut = "Comes out"
        static let stageRead = "Read"
        static let stageQueue = "Save queue"
        static let stageMemory = "Memory"
        // Gap clarity: at-rest payloads and job purposes, not additional status claims.
        static let sessionBackups = "Session backups"
        static let searchAnswers = "Search answers"
        static let nightlyPurpose = "Checks saving, tidies the log"
        static let weeklyPurpose = "Compacts the database"
        static let stageMac = "Local copy"  // R1.10.5 (L50): one name everywhere
        static let stageDrive = "Google Drive"
        static let stageMeaning = "By meaning"
        static let byWordsNotChecked = "by words, not checked"
        static let savedByAgentsInput = "Saved by agents"
        static let answersToAgents = "Answers to agents"
        static func minutesAgoAxis(_ n: Int) -> String { "\(n)\u{00A0}min ago" }
        static func hoursAgoAxis(_ n: Int) -> String { "\(n)\u{00A0}h ago" }
        static func readout(saved: Int) -> String { "\(saved)" }

        // Break phrases
        static let notBeingRead = "not being read"
        static let signedOut = "signed out"
        static func waiting(_ age: String) -> String { "waiting \(age)" }
        static func behind(_ age: String) -> String { "\(age) behind" }
        static func failedNights(_ n: Int) -> String { "failed \(n) nights" }
        static func notConfirmedSince(_ time: String) -> String { "Not confirmed since \(time)" }

        // Fault headlines and details (the pipeline's seven states)
        static func agentNotRead(_ agent: String) -> String { "\(agent)\u{2019}s sessions aren\u{2019}t being read" }
        static let agentNotReadDetail = "Its new work isn\u{2019}t reaching memory. An agent needs to look at this."  // R1.11 (L53)
        static let readerNotReading = "New conversations aren\u{2019}t being read"
        static let driveSignedOut = "Google Drive is signed out"
        static let savesStuck = "Saves are stuck"  // R1.12 (§ C, L53): one wording on every surface
        static func savesWaited(_ count: Int?, _ age: String) -> String {
            count.map { "\($0) saves have waited \(age) and aren\u{2019}t in memory yet." }
                ?? "Saves have waited \(age) and aren\u{2019}t in memory yet."
        }
        static func meaningBehind(_ age: String) -> String { "Search by meaning is \(age) behind" }
        static let meaningBehindUnknown = "Search by meaning is behind"
        static let meaningBehindDetail = "New memories are found by words until it catches up."
        /// R1.12 (§ C, I 6): "in a row" and "failing" go; one sentence on the row, page, dropdown and panel.
        static func nightlyFailed(_ n: Int) -> String { "Nightly check-up failed \(n) nights" }
        static func nightlyFailedShort(_ n: Int) -> String { nightlyFailed(n) }
        /// R1.12 (I 7): the weekly clean's fault, on its own job's streak (L51).
        static func weeklyFailed(_ n: Int) -> String { "Weekly clean failed \(n) weeks" }
        static let notConfirmed = "Not confirmed"
        static func lastCheckWas(_ age: String) -> String { "The last check was \(age) ago, so this may be out of date." }

        // Backups
        static let backupsDetail = "Copies of your memory, on this Mac and in Google Drive."
        static let copied = "copied"
        static let nextRun = "next run"
        static let lateLine = "a copy older than this line is late"
        static let notUploadingSince = "not uploading since"
        static func lateAfter(_ age: String) -> String { "late after \(age)" }
        static func daysAgoAxis(_ n: Int) -> String { "\(n)\u{00A0}d ago" }
        static func inHoursAxis(_ n: Int) -> String { "in \(n)\u{00A0}h" }
        static let newest = "Newest"
        static let next = "Next"
        static let laneMac = "Local copy"  // R1.10.5 (L50)
        static let laneDrive = "Database in Google Drive"  // R1.12 (I 3)
        static let laneConversations = "Sessions in Google Drive"  // R1.12 (I 1)
        /// R1.12 (E1, I 5): the row checks the local COPY (`local_integrity`); the nightly check-up checks memory itself.
        static let laneCheck = "Local copy check"
        static let laneCheckHelp = "Checks that last night\u{2019}s local copy opens and is complete. The nightly check-up checks memory itself."
        static let passed = "passed"
        static let signedIn = "Signed in"
        static let signedOutValue = "Signed out"
        static let newestCopyOnMac = "Newest local copy"  // R1.10.5 (L50)
        static func todayAt(_ hhmm: String) -> String { "Today at \(hhmm)" }
        static let showInFinder = "Show in Finder"
        static let roomForBackups = "Room for backups"
        static let enough = "Enough", tight = "Tight", full = "Full"

        // Jobs
        static let jobsDetail = "The scheduled work that keeps memory read, searchable and backed up."
        static let ran = "ran"
        static let runsAllTheTime = "runs all the time"
        static let everyFewMinutes = "every few minutes"
        static let outsideTimeline = "outside the timeline"
        static let lastRun = "Last run"
        static let nextRunHead = "Next run"
        static let unreadRecord = "Record couldn't be read"
        static func nextRunAt(_ clock: String) -> String { "Next run " + clock }
        static let runsEveryFewMinutes = "Every few minutes."
        static let runsContinuously = "Runs continuously."
        static let scheduleNotKnown = "Schedule not known."
        static func runsDailyAt(_ clock: String) -> String { "Runs daily at \(clock)." }
        static func runsWeeklyAt(_ clock: String) -> String { "Runs weekly at \(clock)." }
        static func runsEvery(_ age: String) -> String { "Runs every \(age)." }
        static func inAge(_ age: String) -> String { "in \(age)" }
        /// Keyed by the job's launchd raw value, so the copy deck stays free of the settings types.
        static let jobNames: [String: String] = [
            // R1.12 (§ C): the stages' names on the Jobs page too ("Read", "Save queue"), and "sessions" for the files.
            "watch": "Read", "hotlane": "Keep search ready", "drain": "Save queue",
            "walCheckpoint": "Tidy up storage", "backupDaily": "Back up the database", "jsonlBackup": "Back up sessions",
            "maintenanceNightly": "Nightly check-up", "maintenanceWeekly": "Weekly clean", "decay": "Fade old memories",
            "repairFTS": "Repair word search",
        ]
        static let jobHelp: [String: String] = [
            "watch": "Reads what your agents write, as they write it.",
            "hotlane": "Keeps search by meaning loaded.",
            "drain": "Writes saves that had to wait.",
            "walCheckpoint": "Folds the write log into the database.",
            "backupDaily": "Copies memory to this Mac, then to Google Drive. Nightly at 03:15.",
            "jsonlBackup": "Copies your agents\u{2019} conversations to Google Drive. Nightly.",
            "maintenanceNightly": "Checks memory can still save, and tidies the log. Nightly at 04:00.",
            "maintenanceWeekly": "Compacts the database. Sundays at 04:00.",
            "decay": "Lowers the weight of memories nobody uses.",
            "repairFTS": "Rebuilds the word index. Weekly.",
        ]

        // Station popovers
        static let databaseCopy = "Database"  // R1.12 (§ C): the Drive panel's two rows
        static let conversationsCopy = "Sessions"
        static let account = "Account"
        static let openBackups = "Open Backups"
        static let openJobs = "Open Jobs"
        static let searchByWordsNotChecked = "Search by words: not checked yet"
        static func running(_ n: Int) -> String { "\(n) running" }

        // R1.1 (CHANGES-R1.1.md; approved with the revision): status lines, line words, the stale state, dropdown details.
        static let conversationsCopiedAsTheyAre = "conversations, copied as they are"
        static let byWordsNotConfirmed = "by words, not confirmed"
        static let everyPipeNotConfirmed = "every pipe: not confirmed"
        static let delayed = "delayed"
        static let stopped = "stopped"
        static let empty = "empty"
        static let working = "working"
        static func percentReady(_ p: Int) -> String { "\(p)% ready" }
        static func nightlyAgo(_ age: String) -> String { "nightly \(age) ago" }
        static func notCheckedSince(_ day: String) -> String { "Memory hasn\u{2019}t been checked since \(day)." }
        static func nothingUploadedFor(_ age: String) -> String { "Nothing has uploaded for \(age)." }
        static func nothingKnownBroken(_ age: String) -> String {
            "Nothing is known to be broken, but nothing has been checked for \(age)."
        }
        static let waitingForNextCheck = "Waiting for the next check"
        // Real-screen round 1 (D6, D7; lead rulings L16/L20 rewritten).
        static let meaningCatchingUp = "Search by meaning is catching up"
        static func lastRead(_ age: String) -> String { "last read \(age) ago" }
        static func lastBackup(_ age: String) -> String { "last backup \(age) ago" }
        static func lastCheckUp(_ age: String) -> String { "last check-up \(age) ago" }
        static func driveAgo(_ age: String) -> String { "Drive \(age) ago" }
        static func macAgo(_ age: String) -> String { "Local copy \(age) ago" }  // R1.12 (§ C): the dropdown's Backups value
        static let lastKnown = "last known"
        /// Each half's scale names the real bucket span, not a bare number that reads like an axis floor.
        static func upTo(_ value: String, minutes: Int) -> String { minutes == 1 ? "up to \(value)/min" : "up to \(value)/\(minutes) min" }
        // Revision 1.2 (L23, L24, L26).
        static let checkSoon = "Check soon"
        static func lastDriveUpload(_ age: String) -> String { "Last Drive upload \(age) ago." }
        static func statusChecked(_ age: String) -> String { "Status checked \(age) ago" }
        static func nextStatusCheck(_ age: String) -> String { "Next status check in \(age)" }
        static func valuesLastKnownAt(_ time: String) -> String { "not confirmed: values last known at \(time)" }
        static func lastKnownAt(_ time: String) -> String { "Last known at \(time)" }
        static let failedThenRetried = "failed, then copied on retry"
        static func uploadFailedRetried(_ failed: String, _ retry: String) -> String {
            "Upload failed at \(failed); the \(retry) retry succeeded."
        }
        static let searchByWordsNoCheck = "Search by words: no check yet"
        static func notCheckedSinceShort(_ day: String) -> String { "Not checked since \(day)." }
        static let nothingKnownBrokenShort = "Nothing is known to be broken."
        static let lateIfLeftOfLine = "late if the newest copy is left of this line"
        static func more(_ n: Int) -> String { "\(n) more" }
        static func plusMore(_ n: Int) -> String { "+\(n) more" }
        static func perBucket(_ value: String, minutes: Int) -> String { minutes == 1 ? "\(value) / min" : "\(value) / \(minutes) min" }

        // Revision 1.3 (L32, L33): the loop's lanes, the short action names, the fault popovers' conditional lines, and the
        // eight station explanations (CHANGES-R1.3 § F, each sentence true of BrainLayer as written).
        static let lanesSessions = "sessions"
        static let lanesSaves = "saves"
        static let sessionFilesCopied = "session files, copied as they are"
        static let answersToAgentsLane = "answers to agents"
        static let yourAgents = "Your agents"
        static func byMeaningBehind(_ age: String) -> String { "by meaning \(age) behind" }
        static let copyReportShort = "Copy report"
        static func lastGoodRun(_ day: String, _ time: String) -> String { "Last good run: \(day), \(time)." }
        static let memoryStillSaves = "Memory still saves and answers searches; it hasn\u{2019}t been checked or tidied since."
        static let notCheckedOrTidiedSince = "It hasn\u{2019}t been checked or tidied since."
        static let macCopyStillMade = "The local copy is still made every night."
        static let explainAgents = "Your coding agents. BrainBar reads the session files they write, and takes the memories they save. Search answers go back to them."
        static let explainRead = "Watches the agents\u{2019} session files as they grow, keeps what matters and turns it into memories."
        static let explainQueue = "Holds memories an agent has saved until they are written. It is normally empty; a save that waits too long shows here."
        static let explainMemory = "The database every memory lives in. Everything else reads from it or copies it."
        static let explainSearch = "Finds memories two ways: by meaning, and by matching words. Agents ask it, and the answers go back to them."
        static let explainMac = "Every night, a checked copy of the database is saved on this Mac."
        static let explainDrive = "The local copy is uploaded to your Google Drive, and the agents\u{2019} session files are copied there too. If Drive is signed out, both stop."
        static let explainJobs = "Read memory, check it and tidy it in place: a light check-up every night and a deep clean that compacts the whole database every week. They add no new memories."

        // Revision 1.4 (L34, L35, L38): provider names, the words that moved into hovers, the readout's first line.
        static let antigravity = "Antigravity"
        static func providerNotRead(_ provider: String) -> String { "\(provider) not read" }
        static func busiestMinute(_ span: String, saved: String, read: String) -> String {
            "Busiest minute \(span): \(saved) saved, \(read) read"
        }
        /// L39: on 3 h and 24 h a bar spans 3 or 24 minutes, and the readout says so.
        static func busiestMinutes(_ span: Int, saved: String, read: String) -> String {
            "Busiest \(span) minutes: \(saved) saved, \(read) read"
        }
        static let thisHour = "this hour"
        static let theseThreeHours = "these 3 hours"
        static let today = "today"
        static let waitingRow = "Waiting"
        static let none = "none"
        static let lastCheck = "Last check"
        static let railTitle = "Session files, copied as they are"
        static let railLine = "The agents\u{2019} session files go to your Google Drive as they are."
        static let answersTitle = "Answers to agents"
        static let answersLine = "What Search finds goes back to the agent that asked."
        static let agentsLine = "Claude, Codex, Cursor and Antigravity."
        static let keptSafeLine = "A local copy every night, and a copy in your Google Drive."
        // Round 2a decisions (R2A-DECISIONS.md, taken 3, 6, 8, 9).
        static func lastUpload(_ age: String) -> String { "last upload \(age) ago" }
        static func readAt(_ time: String) -> String { "read at \(time)" }
        static func backupAt(_ time: String) -> String { "backup at \(time)" }
        static func memoriesCount(_ n: String) -> String { "\(n) \(memoriesWord)" }
        static let memoriesWord = "memories"
        static func stationSays(_ station: String, _ status: String) -> String { "\(station): \(status)" }

        // Revision 1.5 (L45): the Jobs name, the copies' truth, the panel rows, the copy click, the agent panel.
        static let jobsStationName = "Nightly check-up"  // R1.10 (L49): the jobs are two stations now
        static let jobsPanelTitle = "Jobs that work on memory"
        static let jobsPanelLine = "They read memory, check it and tidy it in place. They add no new memories."
        static func keptNewest(_ n: Int, _ age: String) -> String { "\(n) kept, newest \(age) ago" }
        static let keptRow = "Kept"
        static let newestRow = "Newest"
        static let copiedLabel = "Copied"
        static func othersStillRead(_ others: String) -> String {
            "Its new work isn\u{2019}t reaching memory. \(others) are still read."
        }
        /// The keep setting as a word ("three"), for the copy panel's sentence.
        static func countWord(_ n: Int) -> String {
            ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"].indices.contains(n)
                ? ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][n] : "\(n)"
        }

        // Round 2 decisions (R2-DECISIONS "Taken", approved copy).
        static func noDataSince(_ time: String) -> String { "no data since \(time)" }
        // Revision 1.10 (L48 to L51), approved in CHANGES-R1.10 § D. R1.12 (§ C) retires the rows' own short words: each
        // row says its fault's one sentence.
        static let oneThingNeedsYou = "1 thing needs you"
        static let stationNightly = "Nightly check-up"
        static let stationWeekly = "Weekly clean"
        static func lastRunAgo(_ age: String) -> String { "last run \(age) ago" }
        static func failedWeeks(_ n: Int) -> String { "failed \(n) weeks" }
        static let explainNightly = "A light check of memory every night: it checks that saving still works and tidies the write log."
        static let explainWeekly = "Compacts the whole database once a week. It adds no new memories."
        static let localCopy = "Local copy"
        static func dayTodayAt(_ time: String) -> String { "today at \(time)" }
        static func dayYesterdayAt(_ time: String) -> String { "yesterday at \(time)" }
        static func dayWeekdayAt(_ day: String, _ time: String) -> String { "\(day) at \(time)" }
        static let localCopyStillMade = "The local copy is still made every night."

        // Revision 1.12 (L54, CHANGES-R1.12 § I; the 12 approved strings and § C's table).
        /// F: Save queue's status: "stuck {t}" (red), "{n} waiting" under its threshold, "empty".
        static func stuckFor(_ age: String) -> String { "stuck \(age)" }
        static func nWaiting(_ n: Int) -> String { "\(n) waiting" }
        /// F: Search's status, its panel row and the dropdown value.
        static func searchablePercent(_ p: Int) -> String { "\(p)% searchable" }
        static let catchingUp = "catching up"
        static func failedChecks(_ n: Int) -> String { "failed \(n) checks" }
        /// F (I 10): the Search panel; its second sentence only while the `search` check is green.
        static let searchFinds = "Finds memories by meaning and by matching words."
        static let searchLastWorked = "Its last check worked."
        /// E3 (I 11): the Drive panel while the local copy is late: nothing new was made to upload.
        static let nothingNewToUpload = "Nothing new to upload: the local copy is late."
        /// I 12: the Local copy panel, shorter (the same facts as L45).
        static func localCopyPanel(_ keep: String) -> String {
            "Saved every night. The \(keep) newest are kept until Google Drive holds a checked copy."
        }
        /// E3 (L53 C4): the late copy's second line; the next run only when the schedule says.
        static func localLateLine(_ age: String, _ due: String?) -> String {
            due.map { "The newest is \(age) old; the next one is due \($0)." } ?? "The newest is \(age) old."
        }
        static func tonightAt(_ time: String) -> String { "tonight at \(time)" }
        /// E2: last-known values as clock times (in `t3`).
        static func lastReadClock(_ time: String) -> String { "last read \(time)" }
        static func keptNewestClock(_ n: Int, _ time: String) -> String { "\(n) kept, newest \(time)" }
        static func lastUploadClock(_ time: String) -> String { "last upload \(time)" }
        static func lastRunClock(_ time: String) -> String { "last run \(time)" }
        /// R1.10.1: Google Drive's station status while it waits for the browser: the approved word "Waiting" (the row
        /// label), lower case as every station status is.
        static let stationWaiting = "waiting"

        // Settings
        static let settingsTitle = "BrainBar Settings"
        static let reconnectShort = "Reconnect"
        static let settingsFile = "Settings file"

        /// V1's VoiceOver sentence for the waveform.
        static func wave(span minutes: Int, saves: Int, reads: Int) -> String {
            let span = minutes == 60 ? "Last hour" : minutes == 180 ? "Last 3 hours" : "Last 24 hours"
            return "\(span): \(saves) saved by agents, \(reads) session lines ingested."
        }
    }
}

// L62: Drive backup rule Settings, approved 23 Nov. Stable ids live beside every new string.
extension BrainBarMenuPresenter {
    enum DriveRetentionSay {
        static let title = "Google Drive backups" // DR04
        static let purpose = "Each Mac's database backups on Google Drive." // DR05
        static let rule = "Rule" // DR06
        static let daily = "Daily copies" // DR07
        static let monthly = "Monthly copies" // DR08
        static let onDrive = "On Drive now" // DR13
        static let cleanup = "Cleanup" // DR22
        static let removeOld = "Remove old copies" // DR23
        static let cleanupOff = "Cleanup is off: nothing on Drive is removed until you turn it on." // DR24
        static let turnOn = "Turn on cleanup" // DR28
        static let showDetails = "Google Drive backup rule. Show details" // DR03
        static let notReported = "The backup rule is not reported yet. Cleanup is off." // DR29
        static let notMeasured = "Not measured yet" // DR30
        static let lastCleanupNotReported = "Last cleanup is not reported yet." // DR35
        static let trash = "You can restore copies from Drive's trash for 30 days." // DR27
        static let done = "Done"
        static let cancel = "Cancel"
        static func atRest(_ daily: Int, _ monthly: Int) -> String { // DR01
            "\(daily) daily and \(monthly) monthly copies per Mac. Cleanup is off."
        }
        static func explanation(_ daily: Int, _ monthly: Int) -> String { // DR12
            "Daily keeps the newest copy of each of the last \(daily) days. Monthly keeps the last daily copy of each of the \(monthly) previous full months."
        }
        static func summary(_ count: Int, _ size: String) -> String { // DR15
            "\(count) \(count == 1 ? "copy" : "copies"), \(size)"
        }
        static func dateRange(_ oldest: String, _ newest: String) -> String { "\(oldest) to \(newest)" } // DR31
        static func keeps(_ count: Int, _ size: String, _ newest: String?) -> String { // DR16
            "Keeps " + summary(count, size) + (newest.map { ", newest " + $0 } ?? "")
        }
        static let keepsNotMeasured = "Keeps: Not measured yet" // DR36
        static func unverified(_ count: Int, _ size: String) -> String { // DR37
            "Not verified, kept for now: " + summary(count, size)
        }
        static func couldNotMeasure(_ reason: String) -> String { "Couldn't measure: \(reason)" } // DR38
        static func pinned(_ count: Int, _ size: String) -> String { // DR39
            "Pinned, kept for now: " + summary(count, size)
        }
        static func wouldRemove(_ summary: String) -> String { "Would remove: \(summary)" } // DR18
        static func monthlyDates(_ dates: String) -> String { "Monthly: \(dates)" } // DR32
        static func firstCleanup(_ count: Int, _ size: String) -> String { // DR20
            "The first cleanup would move \(count) \(count == 1 ? "copy" : "copies"), \(size), to Drive's trash."
        }
        static func confirm(_ count: Int, _ size: String) -> String { // DR26
            "Move \(count) \(count == 1 ? "copy" : "copies"), \(size), to Drive's trash?"
        }
        static func configError(_ reason: String) -> String { "Backup rule unavailable: \(reason)" } // DR33
        static func stale(_ selectedAt: String, _ error: String?) -> String { // DR34
            "Out of date, last measured \(selectedAt)." + (error.map { " " + $0 } ?? "")
        }
    }
}
