import AppKit
import Darwin
import Foundation

enum BrainBarAppSupport {
    private static let daemonPIDFile = "/tmp/brainbar-daemon.pid"
    private static let daemonLaunchdLabels = [
        "com.brainlayer.brainbar-daemon",
        "com.brainlayer.BrainBarDaemon",
    ]
    private static let daemonBundleIdentifiers = [
        "com.brainlayer.brainbar-daemon",
        "com.brainlayer.BrainBarDaemon",
    ]

    static func hotkeyPermissionFailureMessage(permissions: HotkeyPermissionStatus) -> String {
        "BrainBar could not start the fallback hotkey listener. Enable \(permissions.missingPermissionsMessage) in System Settings. The CGEventTap fallback requires both Input Monitoring and Accessibility."
    }

    @MainActor
    static func makeStatsCollector(
        dbPath: String,
        targetPID: pid_t,
        brainBusEvents: BrainBusEventSource? = BrainBusClient(),
        databaseOpenConfiguration: BrainDatabase.OpenConfiguration = BrainDatabase.OpenConfiguration()
    ) -> StatsCollector {
        StatsCollector(
            dbPath: dbPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: targetPID),
            brainBusEvents: brainBusEvents,
            databaseOpenConfiguration: databaseOpenConfiguration
        )
    }

    @MainActor
    static func makeUIStatsCollector(
        dbPath: String,
        brainBusEvents: BrainBusEventSource? = BrainBusClient(),
        daemonPIDProvider: () -> pid_t = BrainBarAppSupport.discoverDaemonPID
    ) -> StatsCollector {
        makeStatsCollector(
            dbPath: dbPath,
            targetPID: daemonPIDProvider(),
            brainBusEvents: brainBusEvents,
            databaseOpenConfiguration: BrainDatabase.OpenConfiguration(readOnly: true)
        )
    }

    static func discoverDaemonPID() -> pid_t {
        for label in daemonLaunchdLabels {
            if let pid = launchctlPID(for: label) {
                return pid
            }
        }
        if let pid = pidFromFile(daemonPIDFile) {
            return pid
        }
        if let pid = runningApplicationPID() {
            return pid
        }
        return 0
    }

    private static func launchctlPID(for label: String) -> pid_t? {
        let domains = [
            "user/\(getuid())/\(label)",
            "gui/\(getuid())/\(label)",
        ]

        for domain in domains {
            guard let output = runLaunchctlPrint(domain),
                  let pid = parsePID(fromLaunchctlOutput: output)
            else {
                continue
            }
            return pid
        }
        return nil
    }

    private static func runLaunchctlPrint(_ domain: String) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["print", domain]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return nil
        }

        guard process.terminationStatus == 0 else { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)
    }

    private static func parsePID(fromLaunchctlOutput output: String) -> pid_t? {
        for line in output.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.lowercased().hasPrefix("pid") else { continue }

            let tokens = trimmed.components(separatedBy: CharacterSet.decimalDigits.inverted)
                .filter { !$0.isEmpty }
            for token in tokens {
                guard let rawPID = Int32(token), rawPID > 0 else { continue }
                return pid_t(rawPID)
            }
        }
        return nil
    }

    private static func pidFromFile(_ path: String) -> pid_t? {
        guard let contents = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
        let token = contents.trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: CharacterSet.whitespacesAndNewlines)
            .first ?? ""
        guard let rawPID = Int32(token), rawPID > 0 else { return nil }
        return pid_t(rawPID)
    }

    private static func runningApplicationPID() -> pid_t? {
        for bundleIdentifier in daemonBundleIdentifiers {
            if let app = NSRunningApplication.runningApplications(withBundleIdentifier: bundleIdentifier).first {
                return app.processIdentifier
            }
        }

        return NSWorkspace.shared.runningApplications.first { app in
            app.localizedName == "BrainBarDaemon" ||
                app.executableURL?.lastPathComponent == "BrainBarDaemon"
        }?.processIdentifier
    }
}
