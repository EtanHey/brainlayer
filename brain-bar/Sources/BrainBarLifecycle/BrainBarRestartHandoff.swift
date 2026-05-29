import Darwin
import Foundation

public enum BrainBarRestartHandoff {
    public static let markerPath = "/tmp/brainbar-restart-handoff"

    public static func markRestartingProcess(
        pid: pid_t = ProcessInfo.processInfo.processIdentifier,
        at date: Date = Date(),
        path: String = markerPath
    ) {
        let payload = "\(pid)\n\(date.timeIntervalSince1970)\n"
        try? payload.write(toFile: path, atomically: true, encoding: .utf8)
    }

    public static func clear(path: String = markerPath) {
        try? FileManager.default.removeItem(atPath: path)
    }

    public static func consumeIfMatches(
        existingPID: pid_t,
        now: Date = Date(),
        maxAge: TimeInterval = 10,
        path: String = markerPath
    ) -> Bool {
        guard let payload = try? String(contentsOfFile: path, encoding: .utf8) else {
            return false
        }
        let lines = payload
            .split(separator: "\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard lines.count >= 2,
              let markerPID = Int32(String(lines[0])),
              let timestamp = TimeInterval(String(lines[1]))
        else {
            clear(path: path)
            return false
        }

        guard markerPID == existingPID, now.timeIntervalSince1970 - timestamp <= maxAge else {
            if now.timeIntervalSince1970 - timestamp > maxAge {
                clear(path: path)
            }
            return false
        }

        clear(path: path)
        return true
    }
}
