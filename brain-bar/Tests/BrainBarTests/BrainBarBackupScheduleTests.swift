import XCTest
@testable import BrainBar

/// #968: every backup states its schedule from the installed LaunchAgent, its last run from its own
/// log, its next run, and offers Reveal in Finder / Copy path for the latest local copy.
final class BrainBarBackupScheduleTests: XCTestCase {
    private func plist(_ body: String) -> Data {
        Data("""
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0"><dict><key>Label</key><string>x</string>\(body)</dict></plist>
        """.utf8)
    }

    private func calendar(_ zone: String) -> Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: zone)!
        return calendar
    }

    private func date(_ iso: String) -> Date { ISO8601DateFormatter().date(from: iso)! }

    // MARK: cadence text for each schedule shape

    func testCadenceTextForEachScheduleShape() {
        let daily = BackupScheduleRead.parse(
            plist: plist("<key>StartCalendarInterval</key><dict><key>Hour</key><integer>3</integer><key>Minute</key><integer>17</integer></dict>"),
            path: "/LA/daily.plist"
        )
        XCTAssertEqual(daily.text, "daily at 03:17")
        for sunday in [0, 7] {
            let weekly = BackupScheduleRead.parse(
                plist: plist("<key>StartCalendarInterval</key><dict><key>Weekday</key><integer>\(sunday)</integer><key>Hour</key><integer>4</integer><key>Minute</key><integer>0</integer></dict>"),
                path: "/LA/weekly.plist"
            )
            XCTAssertEqual(weekly.text, "weekly on Sunday at 04:00", "launchd Weekday \(sunday)")
        }
        let twice = BackupScheduleRead.parse(
            plist: plist("<key>StartCalendarInterval</key><array><dict><key>Hour</key><integer>3</integer><key>Minute</key><integer>17</integer></dict><dict><key>Hour</key><integer>15</integer><key>Minute</key><integer>17</integer></dict></array>"),
            path: "/LA/twice.plist"
        )
        XCTAssertEqual(twice.text, "daily at 03:17 and daily at 15:17")
        XCTAssertEqual(BackupScheduleRead.parse(plist: plist("<key>StartInterval</key><integer>21600</integer>"), path: "p").text, "every 6 hours")
        XCTAssertEqual(BackupScheduleRead.parse(plist: plist("<key>StartInterval</key><integer>300</integer>"), path: "p").text, "every 5 minutes")
    }

    func testMissingOrUnparseablePlistIsAnHonestUnknown() {
        XCTAssertEqual(
            BackupScheduleRead.parse(plist: nil, path: "/LA/com.brainlayer.backup-daily.plist").text,
            "Schedule unknown — no LaunchAgent installed at /LA/com.brainlayer.backup-daily.plist"
        )
        XCTAssertEqual(
            BackupScheduleRead.parse(plist: Data("garbage".utf8), path: "/LA/x.plist").text,
            "Schedule unknown — /LA/x.plist is not a readable plist"
        )
        XCTAssertEqual(
            BackupScheduleRead.parse(plist: plist("<key>RunAtLoad</key><true/>"), path: "/LA/x.plist").text,
            "Schedule unknown — /LA/x.plist has no StartCalendarInterval or StartInterval"
        )
        XCTAssertNil(BackupScheduleRead.unknown(reason: "x").nextRun(after: Date(), calendar: calendar("UTC")))
    }

    // MARK: next run, including DST and the Sunday-weekly edge

    func testNextRunForDailyAndTheSundayWeeklyEdge() {
        let utc = calendar("UTC")
        XCTAssertEqual(
            BackupCadence.calendar(hour: 3, minute: 17, weekday: nil).nextRun(after: date("2026-09-29T10:00:00Z"), calendar: utc),
            date("2026-09-30T03:17:00Z")
        )
        let sunday = BackupCadence.calendar(hour: 4, minute: 0, weekday: 0)
        // 2026-09-27 is a Sunday: a minute before, it runs today; a minute after, next Sunday.
        XCTAssertEqual(sunday.nextRun(after: date("2026-09-27T03:59:00Z"), calendar: utc), date("2026-09-27T04:00:00Z"))
        XCTAssertEqual(sunday.nextRun(after: date("2026-09-27T04:01:00Z"), calendar: utc), date("2026-10-04T04:00:00Z"))
        XCTAssertEqual(
            BackupCadence.interval(seconds: 3_600).nextRun(after: date("2026-09-29T10:00:00Z"), calendar: utc),
            nil,
            "an interval job's next run depends on launchd's own clock, so it is not guessed"
        )
    }

    func testNextRunAcrossDaylightSavingTransitions() {
        let newYork = calendar("America/New_York")
        // Fall back (2026-11-01): 03:17 happens once, in EST.
        let fallBack = BackupCadence.calendar(hour: 3, minute: 17, weekday: nil)
            .nextRun(after: date("2026-10-31T16:00:00Z"), calendar: newYork)
        XCTAssertEqual(fallBack, date("2026-11-01T08:17:00Z"))
        // Spring forward (2026-03-08): 02:30 does not exist; it runs at the next existing time that day.
        let springForward = BackupCadence.calendar(hour: 2, minute: 30, weekday: nil)
            .nextRun(after: date("2026-03-07T17:00:00Z"), calendar: newYork)
        let parts = newYork.dateComponents([.day, .hour], from: springForward!)
        XCTAssertEqual(parts.day, 8)
        XCTAssertEqual(parts.hour, 3)
    }

    // MARK: last run from the real logs, never a guess

    func testLastRunComesFromRealLogRowsOnly() {
        let dbLog = Data("""
        {"attempted_at": "2026-09-28T00:17:05+00:00", "verified": true, "backup_log_provenance": "real"}
        {"attempted_at": "2026-09-29T00:17:06.113369+00:00", "verified": true, "backup_log_provenance": "real"}
        {"attempted_at": "2026-09-29T09:00:00+00:00", "verified": true, "backup_log_provenance": "pytest"}
        not json
        """.utf8)
        let dbRun = BackupLogReader.lastRun(.databaseBackup, log: dbLog)
        XCTAssertEqual(dbRun?.at.timeIntervalSince1970 ?? 0, date("2026-09-29T00:17:06Z").timeIntervalSince1970 + 0.113, accuracy: 0.001)
        XCTAssertEqual(dbRun?.verified, true, "the pytest-provenance row after it is skipped")
        let archiveLog = Data("""
        {"attempted_at": "2026-09-29T02:01:00+00:00", "verified": false, "status": "failed"}
        """.utf8)
        XCTAssertEqual(BackupLogReader.lastRun(.transcriptArchive, log: archiveLog)?.verified, false)
        let maintenanceLog = Data("""
        {"ts": "2026-09-27T01:30:00+00:00", "mode": "full", "dry_run": false, "vacuum_after_bytes": 15926001664}
        {"ts": "2026-09-28T01:05:00+00:00", "mode": "full", "backup_status": "waiting"}
        {"ts": "2026-09-29T01:05:00+00:00", "mode": "light", "dry_run": false}
        {"ts": "2026-09-29T02:00:00+00:00", "mode": "full", "dry_run": true}
        """.utf8)
        XCTAssertEqual(
            BackupLogReader.lastRun(.weeklyMaintenance, log: maintenanceLog),
            BackupRunReceipt(at: date("2026-09-27T01:30:00Z"), verified: nil),
            "only a completed, non-dry-run full pass counts"
        )
        XCTAssertNil(BackupLogReader.lastRun(.databaseBackup, log: nil))
    }

    // MARK: latest local copies

    func testLatestLocalCopiesIgnoreNonSnapshotFiles() {
        XCTAssertEqual(
            BackupLocalFiles.latestSnapshot(in: [
                ".backup.lock", "2026-09-21.db.gz", "2026-09-23.db.gz", "2026-09-29.db",
                "pre-redact-2026-09-30.db", "pre-redact-2026-09-30.db.complete", "2026-09-30.db.gz.tmp",
            ]),
            "2026-09-29.db"
        )
        XCTAssertEqual(
            BackupLocalFiles.latestArchive(in: [
                "claude-jsonl-2026-09-28.tar.gz", "claude-jsonl-2026-09-29.tar.gz",
                ".claude-jsonl-2026-09-30.tar.gz.x.tmp", "notes.txt",
            ]),
            "claude-jsonl-2026-09-29.tar.gz"
        )
        XCTAssertNil(BackupLocalFiles.latestSnapshot(in: [".backup.lock"]))
    }

    // MARK: #1015 review round 1

    private func calendarPlist(_ entry: String) -> Data {
        plist("<key>StartCalendarInterval</key><dict>\(entry)</dict>")
    }

    private func int(_ key: String, _ value: Int) -> String { "<key>\(key)</key><integer>\(value)</integer>" }

    /// B1: only a completed full pass is a "last run". A final row whose backup was unavailable or
    /// whose post-backup gate failed aborted before VACUUM; it is reported as an attempt, not a run.
    func testAbortedMaintenancePassIsNeverTheLastRun() {
        let log = Data("""
        {"ts": "2026-08-30T02:23:03+00:00", "mode": "full", "dry_run": false, "vacuum_after_bytes": 15926001664}
        {"ts": "2026-09-20T01:10:00+00:00", "mode": "full", "dry_run": false, "backup_status": "gates_failed", "vacuum_after_bytes": null}
        {"ts": "2026-09-27T01:40:00+00:00", "mode": "full", "dry_run": false, "backup_status": "unavailable", "vacuum_after_bytes": null}
        """.utf8)
        XCTAssertEqual(
            BackupLogReader.lastRun(.weeklyMaintenance, log: log),
            BackupRunReceipt(at: date("2026-08-30T02:23:03Z"), verified: nil)
        )
        XCTAssertEqual(
            BackupLogReader.lastIncompleteAttempt(.weeklyMaintenance, log: log),
            BackupIncompleteAttempt(at: date("2026-09-27T01:40:00Z"), reason: "backup unavailable, VACUUM skipped")
        )
        let completedLater = log + Data(("\n" + #"{"ts": "2026-10-04T01:30:00+00:00", "mode": "full", "dry_run": false, "backup_status": "verified", "vacuum_after_bytes": 1}"#).utf8)
        XCTAssertEqual(BackupLogReader.lastRun(.weeklyMaintenance, log: completedLater)?.at, date("2026-10-04T01:30:00Z"))
        XCTAssertNil(
            BackupLogReader.lastIncompleteAttempt(.weeklyMaintenance, log: completedLater),
            "an attempt older than the last completed pass is history, not news"
        )
    }

    /// B2: out-of-range or non-integer calendar values never trap and never produce false text.
    func testInvalidCalendarValuesAreAnHonestUnknown() {
        let cases: [(String, String)] = [
            (int("Weekday", -1) + int("Hour", 4) + int("Minute", 0), "invalid Weekday -1"),
            (int("Weekday", 8) + int("Hour", 4) + int("Minute", 0), "invalid Weekday 8"),
            (int("Hour", 24) + int("Minute", 0), "invalid Hour 24"),
            (int("Hour", 3) + int("Minute", 60), "invalid Minute 60"),
            ("<key>Hour</key><string>3</string>" + int("Minute", 17), "non-integer Hour"),
        ]
        for (entry, reason) in cases {
            let read = BackupScheduleRead.parse(plist: calendarPlist(entry), path: "/LA/x.plist")
            XCTAssertEqual(read.text, "Schedule unknown — /LA/x.plist has \(reason)", entry)
            XCTAssertNil(read.nextRun(after: date("2026-09-29T10:00:00Z"), calendar: calendar("UTC")))
        }
    }

    /// B3: every launchd calendar shape is rendered faithfully or reported as not representable.
    func testEveryLaunchdCalendarShape() {
        let utc = calendar("UTC")
        let after = date("2026-09-29T10:00:00Z")
        func read(_ entry: String) -> BackupScheduleRead { BackupScheduleRead.parse(plist: calendarPlist(entry), path: "/LA/x.plist") }

        let hourly = read(int("Minute", 5))
        XCTAssertEqual(hourly.text, "hourly at :05")
        XCTAssertEqual(hourly.nextRun(after: after, calendar: utc), date("2026-09-29T10:05:00Z"))

        let monthly = read(int("Day", 1) + int("Hour", 3) + int("Minute", 17))
        XCTAssertEqual(monthly.text, "monthly on day 1 at 03:17")
        XCTAssertEqual(monthly.nextRun(after: after, calendar: utc), date("2026-10-01T03:17:00Z"))

        let yearly = read(int("Month", 1) + int("Day", 1) + int("Hour", 3) + int("Minute", 17))
        XCTAssertEqual(yearly.text, "yearly on January 1 at 03:17")
        XCTAssertEqual(yearly.nextRun(after: after, calendar: utc), date("2027-01-01T03:17:00Z"))

        // launchd fires when EITHER Day or Weekday matches; a single calendar match can't express that.
        XCTAssertEqual(
            read(int("Day", 1) + int("Weekday", 0) + int("Hour", 3) + int("Minute", 17)).text,
            "Schedule unknown — /LA/x.plist has a schedule that is not representable (Day with Weekday)"
        )
        XCTAssertEqual(
            read(int("Month", 1) + int("Hour", 3) + int("Minute", 17)).text,
            "Schedule unknown — /LA/x.plist has a schedule that is not representable (Month without Day)"
        )
        XCTAssertEqual(
            read(int("Hour", 3)).text,
            "Schedule unknown — /LA/x.plist has a schedule that is not representable (no Minute: every minute)"
        )

        // launchd evaluates StartCalendarInterval and StartInterval independently: show both.
        let mixed = BackupScheduleRead.parse(
            plist: plist(
                "<key>StartCalendarInterval</key><dict>\(int("Hour", 3) + int("Minute", 17))</dict><key>StartInterval</key><integer>21600</integer>"
            ),
            path: "/LA/x.plist"
        )
        XCTAssertEqual(mixed.text, "daily at 03:17 and every 6 hours")
        XCTAssertNil(mixed.nextRun(after: after, calendar: utc), "the interval part can fire sooner; its clock is launchd's")

        let badArray = BackupScheduleRead.parse(
            plist: plist(
                "<key>StartCalendarInterval</key><array><dict>\(int("Hour", 3) + int("Minute", 17))</dict><dict>\(int("Hour", 25) + int("Minute", 0))</dict></array>"
            ),
            path: "/LA/x.plist"
        )
        XCTAssertEqual(badArray.text, "Schedule unknown — /LA/x.plist has invalid Hour 25 (entry 2)")
    }


    // MARK: #1015 review round 2

    /// B4: launchd fires a Day/Month schedule only on dates that exist. A monthly Day=31 skips
    /// short months and a yearly February 29 waits for a leap year: never a rolled-over March 1.
    func testMonthlyAndYearlyNextRunsOnlyOnDatesThatExist() {
        let utc = calendar("UTC")
        func read(_ entry: String) -> BackupScheduleRead { BackupScheduleRead.parse(plist: calendarPlist(entry), path: "/LA/x.plist") }

        let day31 = read(int("Day", 31) + int("Hour", 3) + int("Minute", 17))
        XCTAssertEqual(day31.text, "monthly on day 31 at 03:17")
        XCTAssertEqual(day31.nextRun(after: date("2026-01-31T04:00:00Z"), calendar: utc), date("2026-03-31T03:17:00Z"))
        XCTAssertEqual(day31.nextRun(after: date("2026-01-31T02:00:00Z"), calendar: utc), date("2026-01-31T03:17:00Z"))
        XCTAssertEqual(day31.nextRun(after: date("2026-03-31T04:00:00Z"), calendar: utc), date("2026-05-31T03:17:00Z"))

        let day30 = read(int("Day", 30) + int("Hour", 3) + int("Minute", 17))
        XCTAssertEqual(day30.nextRun(after: date("2026-02-01T00:00:00Z"), calendar: utc), date("2026-03-30T03:17:00Z"))

        let leapDay = read(int("Month", 2) + int("Day", 29) + int("Hour", 3) + int("Minute", 17))
        XCTAssertEqual(leapDay.text, "yearly on February 29 at 03:17")
        XCTAssertEqual(leapDay.nextRun(after: date("2026-01-15T00:00:00Z"), calendar: utc), date("2028-02-29T03:17:00Z"))

        // A date that never occurs never runs: Unknown with the reason, and no next-run claim.
        for (month, day, name) in [(2, 30, "February 30"), (2, 31, "February 31"), (4, 31, "April 31")] {
            let never = read(int("Month", month) + int("Day", day) + int("Hour", 3) + int("Minute", 17))
            XCTAssertEqual(never.text, "Schedule unknown — /LA/x.plist has a date that never occurs (\(name))")
            XCTAssertNil(never.nextRun(after: date("2026-01-15T00:00:00Z"), calendar: utc))
        }
    }

    /// B4: exact date matching keeps the clock-only DST behaviour: a skipped time runs at the next
    /// existing time that day, and a repeated time runs once, at its first occurrence.
    func testMonthlyNextRunAcrossDaylightSavingTransitions() {
        let newYork = calendar("America/New_York")
        // Spring forward (2026-03-08): 02:30 does not exist.
        let spring = BackupCadence.calendar(hour: 2, minute: 30, day: 8)
            .nextRun(after: date("2026-03-01T12:00:00Z"), calendar: newYork)
        let parts = newYork.dateComponents([.month, .day, .hour], from: spring!)
        XCTAssertEqual([parts.month, parts.day, parts.hour], [3, 8, 3])
        XCTAssertEqual(
            spring,
            BackupCadence.calendar(hour: 2, minute: 30).nextRun(after: date("2026-03-07T17:00:00Z"), calendar: newYork),
            "a monthly run on the DST day lands where the daily run does"
        )
        // Fall back (2026-11-01): 01:30 happens twice; it runs at the first (EDT, UTC-4).
        let fall = BackupCadence.calendar(hour: 1, minute: 30, day: 1)
            .nextRun(after: date("2026-10-15T00:00:00Z"), calendar: newYork)
        XCTAssertEqual(fall, date("2026-11-01T05:30:00Z"))
    }

    /// B2 residual: a plist <true/> bridges to Int 1 through NSNumber. A boolean is not an integer.
    func testPlistBooleansAreNotIntegers() {
        let cases: [(String, String)] = [
            ("<key>Weekday</key><true/>" + int("Hour", 4) + int("Minute", 0), "non-integer Weekday"),
            (int("Hour", 4) + "<key>Minute</key><false/>", "non-integer Minute"),
            ("<key>Day</key><true/>" + int("Hour", 4) + int("Minute", 0), "non-integer Day"),
            ("<key>Hour</key><real>3.5</real>" + int("Minute", 0), "non-integer Hour"),
        ]
        for (entry, reason) in cases {
            let read = BackupScheduleRead.parse(plist: calendarPlist(entry), path: "/LA/x.plist")
            XCTAssertEqual(read.text, "Schedule unknown — /LA/x.plist has \(reason)", entry)
            XCTAssertNil(read.nextRun(after: date("2026-09-29T10:00:00Z"), calendar: calendar("UTC")))
        }
        let interval = BackupScheduleRead.parse(plist: plist("<key>StartInterval</key><true/>"), path: "/LA/x.plist")
        XCTAssertEqual(interval.text, "Schedule unknown — /LA/x.plist has an invalid StartInterval")
    }
}
