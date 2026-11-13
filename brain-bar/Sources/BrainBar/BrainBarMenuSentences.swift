import Foundation

/// The closed list (phase 8, P2b): every sentence, headline, label and standing item the presenter can put in its value,
/// each with a stable id (its truth-table row or contract case), its text or template, a kind, and, for a claim or a
/// reassurance, the condition on the capture under which it is true. The presenter builds every string from these entries
/// and the word formatters below; `finalGuard` checks every sentence it is about to return against them. The invariant
/// test keeps its own conditions under the same ids and fails on any emitted sentence that is not registered here.
extension BrainBarMenuPresenter {
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

        /// The placeholder values when `sentence` is this entry with each slot filled by a word its formatter can make.
        func match(_ sentence: String) -> [String]? {
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
                guard Say.fits(slot, String(value)) else { return nil }
                values.append(String(value))
            }
            return values
        }
    }

    /// The entry and placeholder values for one sentence, or nil when no entry makes it (a fault the guard corrects).
    static func entry(for sentence: String) -> (Sentence, [String])? {
        if let exact = Say.exact[sentence] { return (exact, []) }
        for entry in Say.templated { if let values = entry.match(sentence) { return (entry, values) } }
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
        /// R02's truth: a readable, fresh watcher with a working probe reports lag, `ingest` is present and green, and the
        /// episode is not past the 30-minute limit (R24 escalates it).
        static func catchingUp(_ c: Capture) -> Bool {
            let ingest = c.verdict?.checks.filter { $0.name == "ingest" } ?? []
            return c.watcher.map { $0.readable && $0.fresh && !$0.dbProbeFailed && $0.alertReasons.contains("offset_lag") } == true && !lagPastLimit(c)
                && !ingest.isEmpty && ingest.allSatisfy { $0.state == .green }
        }
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

        // MARK: the calm form (A2)

        static let allGood = S("S1", .reassurance, "All good") { c, _, _ in allGoodProven(c) }
        static let backedUp = S("S01", .claim, "Backed up to Google Drive {at}.") { c, p, v in
            p.headline == allGood.text && allGoodDate(c).map { at($0, c) } == v[0]
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
            progress(c, v[0]) && runningUpload(c)?.onSchedule?.eta.map { calendar(c).isDate($0, inSameDayAs: c.now) && hhmm($0, c) == v[1] } == true
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
        static let fineIs = S("A2.fine.is", .reassurance, "{fine} is fine.") { c, _, v in fineThings(v[0]).isSubset(of: provenFine(c)) }
        static let fineAre = S("A2.fine.are", .reassurance, "{fines} are fine.") { c, _, v in fineThings(v[0]).isSubset(of: provenFine(c)) }
        static let everythingElse = S("M5", .reassurance, "Everything else is fine.") { c, p, _ in uncovered(p).isSubset(of: provenFine(c)) }

        // MARK: the needs form (A1)

        static let countOne = S("A1.count.one", .claim, "1 thing needs you") { _, p, _ in countIs(p, "1") }
        static let countMany = S("A1.count.many", .claim, "{N2} things need you") { _, p, v in countIs(p, v[0]) }
        /// R24: true whenever its row is counted as Needs you (I4), for an amber check or one that cannot be read too.
        static let generic = S("R17", .neutral, "Something BrainBar checks needs attention.")
        static let cantFix = S("U12", .neutral, "BrainBar can’t fix this itself.")
        static let giveReport = S("U12.give", .neutral, "Give the report to an agent.")
        static let ifNotFinished = S("D10", .neutral, "If this backup doesn’t finish, BrainBar can’t fix it itself.")
        static let u03 = S("U03", .claim, "Nightly upkeep failed {N3} nights in a row", .upkeep) { c, _, v in
            c.maintenanceNightFailures >= 3 && String(c.maintenanceNightFailures) == v[0]
        }
        static let stillSaved = S("U03.stillSaved", .reassurance, "Your memory is still being saved and backed up, as of the last checks.") { c, _, _ in
            provenFine(c).isSuperset(of: [.remembering, .safe])
        }
        static let s19 = S("S19", .claim, "The newest checked backup is damaged", .safe) { c, _, _ in red(c, "integrity") }
        static let s19Memory = S("S19.memory", .claim, "Your memory itself may be damaged too.") { c, _, _ in red(c, "integrity") }
        static let f07 = S("F07", .claim, "Search by meaning isn’t catching up", .finding) { c, _, _ in stalled(c) }
        static let f07Since = S("F07.since", .claim, "BrainBar hasn’t seen the memories waiting for it go down{over}.") { c, _, v in
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
            c.backupDailyFailures >= 2 && String(c.backupDailyFailures) == v[0]
        }
        static let s06Newest = S("S06.newest", .claim, "Your newest backup is {age}.") { c, _, v in
            c.backupsMeasured && age(c, ["backup_local", "backup_offsite"], v[0])
        }
        static let s06Confirmed = S("S06.confirmed", .claim, "The newest backup BrainBar can confirm is {age}.") { c, _, v in
            age(c, ["backup_local", "backup_offsite"], v[0])
        }
        static let r15 = S("R15", .claim, "Your newest backup in Google Drive is {age}.", .safe) { c, _, v in c.backupsMeasured && age(c, ["backup_offsite"], v[0]) }
        static let s13 = S("S13", .claim, "Google Drive is disconnected", .safe) { c, _, _ in drive(c, missing: false) }
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
        static let copyReport = S("action.copyReport", .neutral, "Copy report for an agent")
        static let storage = S("action.storage", .neutral, "Open Storage Settings")
        static let settings = S("R4.settings", .neutral, "Settings…")
        static let quit = S("R4.quit", .neutral, "Quit BrainBar")
        static let restart = S("R4.restart", .neutral, "Restart BrainBar")
        static let settingsKey = "⌘,", quitKey = "⌘Q"

        static let all: [S] = [
            allGood, backedUp, checkingMemory, checkingSleep, notConfirmed, lastBackup, r02Headline, r02, s03Headline, s03, s03Started, s03StartedOn,
            s03Progress, s03ProgressEta, catchingTwo, catchingFew, f02, checking, cantConfirmBackups, stillChecking, stillCheckingFew,
            fineIs, fineAre, everythingElse, countOne, countMany, generic, cantFix, giveReport, ifNotFinished, u03, stillSaved, s19,
            s19Memory, f07, f07Since, exactWords, noRecord, nothingInDrive, s32, s32Progress, s32On, s32ProgressOn, s06, s06Newest, s06Confirmed, r15, s13,
            s13Missing, s13Detail, s13DetailMissing, d13, d13Missing, reconnect, connect, copyReport, storage, settings, quit, restart,
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

        /// The nouns a list names, when it is exactly what `joined` (or `list`) makes from distinct nouns in the things' order.
        static func canonical(_ value: String, _ nouns: [Thing: String], capitalised: Bool) -> [String]? {
            let ordered = Thing.allCases.compactMap { nouns[$0] }
            let words = (capitalised ? value.prefix(1).lowercased() + value.dropFirst() : value)
                .replacingOccurrences(of: " and ", with: ", ").components(separatedBy: ", ")
            let at = words.compactMap { ordered.firstIndex(of: $0) }
            guard at.count == words.count, at == at.sorted(), Set(at).count == at.count else { return nil }
            return (capitalised ? list(words) : joined(words)) == value ? words : nil
        }

        /// The things a "… fine." list names.
        static func fineThings(_ value: String) -> Set<Thing> {
            Set(value.lowercased().replacingOccurrences(of: " and ", with: ", ").components(separatedBy: ", ").compactMap { word in
                fineNoun.first { $0.value == word }?.key
            })
        }

        private static let months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        /// Whether `value` is a word its slot's formatter can make (round 1 of #1788, Codex B2): numbers in canonical decimal
        /// within their range (N2 at least 2, N3 at least 3, p 0 to 100), real clock times and days, the age and `{over}`
        /// forms exactly, and noun lists exactly as `joined` / `list` make them.
        static func fits(_ slot: String, _ value: String) -> Bool {
            func number(_ s: some StringProtocol, _ range: ClosedRange<Int>) -> Bool { Int(s).map { String($0) == s && range.contains($0) } == true }
            func clock(_ s: some StringProtocol) -> Bool {
                s.count == 5 && s.dropFirst(2).first == ":" && Int(s.prefix(2)).map { (0...23).contains($0) } == true && s.prefix(2).allSatisfy(\.isNumber)
                    && Int(s.suffix(2)).map { (0...59).contains($0) } == true && s.suffix(2).allSatisfy(\.isNumber)
            }
            func day(_ s: String, relative: Bool) -> Bool {
                let halves = s.components(separatedBy: " at "), date = halves[0].split(separator: " ")
                guard halves.count == 2, clock(halves[1]) else { return false }
                return (relative && ["today", "yesterday"].contains(halves[0])) || (date.count == 2 && months.contains(String(date[0])) && number(date[1], 1...31))
            }
            switch slot {
            case "N2": return number(value, 2...Int.max)
            case "N3": return number(value, 3...Int.max)
            case "p": return number(value, 0...100)
            case "HH:mm": return clock(value)
            case "at": return day(value, relative: true)
            case "age":
                return value == "1 hour old" || (value.hasSuffix(" hours old") && number(value.dropLast(10), 2...23))
                    || (value.hasPrefix("from ") && day(String(value.dropFirst(5)), relative: false))
            case "over":
                let words = value.split(separator: " ", omittingEmptySubsequences: false).map(String.init)
                guard !value.isEmpty else { return true }
                return words.count == 5 && words[0].isEmpty && words[1] == "in" && words[2] == "over"
                    && (words[3...] == ["an", "hour"] || words[3...] == ["a", "minute"] || (number(words[3], 2...Int.max) && words[4] == "hours")
                        || (number(words[3], 2...59) && words[4] == "minutes"))
            case "thing": return behindNoun.values.contains(value)
            case "two": return canonical(value, behindNoun, capitalised: false)?.count == 2
            case "fine": return canonical(value, fineNoun, capitalised: true)?.count == 1
            case "fines": return (canonical(value, fineNoun, capitalised: true)?.count ?? 0) >= 2
            default: return false
            }
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

    static func hhmm(_ date: Date, _ c: Capture) -> String { formatter("HH:mm", c).string(from: date) }

    static func calendar(_ c: Capture) -> Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = c.timeZone
        return calendar
    }

    /// `{at}`, a past event: "today at 04:00", "yesterday at 23:10" (both under 24 h), else "Nov 9 at 03:44".
    static func at(_ date: Date, _ c: Capture) -> String {
        let calendar = calendar(c)
        if c.now.timeIntervalSince(date) < 86_400 {
            if calendar.isDate(date, inSameDayAs: c.now) { return "today at \(hhmm(date, c))" }
            if let yesterday = calendar.date(byAdding: .day, value: -1, to: c.now), calendar.isDate(date, inSameDayAs: yesterday) {
                return "yesterday at \(hhmm(date, c))"
            }
        }
        return formatter("MMM d 'at' HH:mm", c).string(from: date)
    }

    /// A time the words can name: Foundation formats a time outside `distantPast…distantFuture` (or not finite) as a
    /// date that does not exist ("Sep 20 at 02:00" for −∞), so no sentence names one (P2b, found by writing S4's condition).
    static func sayable(_ date: Date) -> Bool { (Date.distantPast...Date.distantFuture).contains(date) }

    static func percent(_ progress: Double) -> Int { Int((progress * 100).rounded()) }

    static func valid(_ age: Double?) -> Bool { age.map { $0.isFinite && $0 >= 0 } == true }

    /// `{copy age}` at `now`: the copy's time is `evaluatedAt − ageS` (wall-clock for backups, U10), and its age is counted
    /// from that time to `now`, never from `evaluatedAt` (a verdict can stay awake-fresh across hours of sleep): "21 hours
    /// old" under a day, else "from Nov 9 at 03:44". Nil for no, a negative or non-finite age, or a time after now.
    static func copyAge(_ age: Double?, _ c: Capture, _ v: BrainBarHealthVerdict) -> String? {
        guard let age, valid(age) else { return nil }
        let copied = v.evaluatedAt.addingTimeInterval(-age), elapsed = c.now.timeIntervalSince(copied)
        guard sayable(copied), elapsed.isFinite, elapsed >= 0 else { return nil }
        guard elapsed >= 86_400 else {
            let hours = min(23, max(1, Int((elapsed / 3600).rounded())))
            return hours == 1 ? "1 hour old" : "\(hours) hours old"
        }
        return "from \(at(copied, c))"
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
