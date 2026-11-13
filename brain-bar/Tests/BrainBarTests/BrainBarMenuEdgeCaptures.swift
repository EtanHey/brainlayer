import Foundation
@testable import BrainBar

/// Captures whose date fields sit on time edges (round 1 of #1790), for the property that every sentence the presenter says
/// is registered for its capture. Each moment: both daylight-saving changes in New York and Berlin (the repeated hour's first
/// and second pass, either side of the skipped hour), a year's last instant, a minute's last instant to a tenth of a
/// millisecond, and one ten years old. Each is said from nows at the 24-hour boundary (−1 ulp, exactly, +1 ulp, ±0.4 ms, a
/// minute inside), at the moment, a minute, an hour and two hours later, two days later and ten years later; as the last
/// known backup, the all-good copy, an upload's start (with an estimate either side of the next midnight), a database copy
/// and a Drive copy.
extension BrainBarMenuInvariantTests {
    static func edgeCaptures() -> [Capture] {
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let ny = TimeZone(identifier: "America/New_York")!, berlin = TimeZone(identifier: "Europe/Berlin")!, plus2 = TimeZone(secondsFromGMT: 7200)!
        let edges: [(String, TimeZone)] = [
            ("2026-11-01T01:50:00.000-04:00", ny), ("2026-11-01T01:50:00.000-05:00", ny), ("2026-03-08T01:59:59.999-05:00", ny),
            ("2026-03-08T03:00:00.000-04:00", ny), ("2026-10-25T02:50:00.000+02:00", berlin), ("2026-10-25T02:50:00.000+01:00", berlin),
            ("2026-03-29T01:59:59.999+01:00", berlin), ("2026-03-29T03:00:00.000+02:00", berlin), ("2026-12-31T23:59:59.999+02:00", plus2),
            ("2026-11-12T12:48:59.999+02:00", plus2), ("2016-11-12T12:48:00.000+02:00", plus2),
        ]
        func at(_ t: Double) -> Date { Date(timeIntervalSinceReferenceDate: t) }
        var out: [Capture] = []
        for (text, zone) in edges {
            for m in [iso.date(from: text)!, iso.date(from: text)!.addingTimeInterval(0.0009)] {
                let day = m.timeIntervalSinceReferenceDate + 86_400
                let nows = [day.nextDown, day, day.nextUp, day - 0.0004, day + 0.0004, day - 59.9995].map(at)
                    + [0, 60, 3600, 7200, 2 * 86_400, 3653 * 86_400].map { m.addingTimeInterval($0) }
                for now in nows {
                    var plain = capture(Input(settings: [:], base: 0, upkeep: 0), dot: .green, checking: false, wake: false, freshness: .fresh0, measured: true)
                    (plain.now, plain.timeZone) = (now, zone)
                    let age = now.timeIntervalSince(m)
                    func with(_ named: [String: (State, Double?)], _ edit: (inout Capture) -> Void = { _ in }) -> Capture {
                        var c = plain
                        c.verdict = BrainBarHealthVerdict(dot: .green, checks: plain.verdict!.checks.map { k in
                            named[k.name].map { Check(name: k.name, state: $0.0, ageS: $0.1, thresholdS: k.thresholdS ?? 280_800, note: "") } ?? k
                        }, notes: [], evaluatedAt: now)
                        edit(&c)
                        return c
                    }
                    var calendar = Calendar(identifier: .gregorian)
                    calendar.timeZone = zone
                    let midnight = calendar.date(byAdding: .day, value: 1, to: calendar.startOfDay(for: now))!
                    out.append(with([:]) { $0.verdict = nil; $0.lastKnownBackupAt = m })  // S4
                    out.append(with(["backup_offsite": (.green, age), "transcripts": (.green, max(0, age - 60))]))  // S01
                    out.append(with(["backup_offsite": (.amber, 27_660)]) { $0.upload = .init(startedAt: m, live: true) })  // S03
                    for eta in [midnight.addingTimeInterval(-0.0001), midnight.addingTimeInterval(0.0001)] {  // S03 with an estimate
                        out.append(with(["backup_offsite": (.amber, 27_660)]) {
                            $0.offsiteRecovering = .init(progress: 0.4, eta: eta, startedAt: m, updatedAt: now, cadenceS: 30)
                        })
                    }
                    out.append(with(["backup_local": (.red, nil), "backup_offsite": (.amber, 27_660)]) { $0.upload = .init(startedAt: m, live: true) })  // S32
                    out.append(with(["drive_auth": (.red, nil), "backup_offsite": (.red, age)]))  // R15 in the Drive row
                    out.append(with(["backup_local": (.red, age), "backup_offsite": (.green, age + 3600)]) { $0.backupDailyFailures = 2 })  // S06
                }
            }
        }
        return out
    }
}
