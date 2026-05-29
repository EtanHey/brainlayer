import AppKit
import Darwin
import Foundation

public final class BrainBarLifecycleWatchdog: @unchecked Sendable {
    public struct Configuration: Sendable {
        let watchedName: String
        let heartbeatPath: String
        let staleTimeout: TimeInterval
        let checkInterval: TimeInterval
        let terminateGraceInterval: TimeInterval
        let relaunchCommand: RelaunchCommand

        public init(
            watchedName: String,
            heartbeatPath: String,
            staleTimeout: TimeInterval = 45,
            checkInterval: TimeInterval = 10,
            terminateGraceInterval: TimeInterval = 2,
            relaunchCommand: RelaunchCommand
        ) {
            self.watchedName = watchedName
            self.heartbeatPath = heartbeatPath
            self.staleTimeout = staleTimeout
            self.checkInterval = checkInterval
            self.terminateGraceInterval = terminateGraceInterval
            self.relaunchCommand = relaunchCommand
        }
    }

    public struct RelaunchCommand: Sendable {
        let executablePath: String
        let arguments: [String]

        public static func launchctlKickstart(label: String) -> RelaunchCommand {
            RelaunchCommand(
                executablePath: "/bin/launchctl",
                arguments: ["kickstart", "-k", "gui/\(getuid())/\(label)"]
            )
        }

        public static func openBundle(_ bundlePath: String) -> RelaunchCommand {
            RelaunchCommand(executablePath: "/usr/bin/open", arguments: ["-n", bundlePath])
        }
    }

    public static let uiHeartbeatPath = "/tmp/brainbar-ui.heartbeat"
    public static let daemonHeartbeatPath = "/tmp/brainbar-daemon.heartbeat"
    public static let uiLaunchAgentLabel = "com.brainlayer.brainbar"
    public static let daemonLaunchAgentLabel = "com.brainlayer.brainbar-daemon"

    private let configuration: Configuration
    private let processProvider: @Sendable () -> [pid_t]
    private let terminateProcess: @Sendable (pid_t, Int32) -> Void
    private let relaunch: @Sendable (RelaunchCommand) -> Void
    private let queue = DispatchQueue(label: "com.brainlayer.brainbar.lifecycle-watchdog", qos: .utility)
    private var timer: DispatchSourceTimer?
    private var isRestarting = false

    init(
        configuration: Configuration,
        processProvider: @escaping @Sendable () -> [pid_t],
        terminateProcess: @escaping @Sendable (pid_t, Int32) -> Void = { pid, signal in
            _ = Darwin.kill(pid, signal)
        },
        relaunch: @escaping @Sendable (RelaunchCommand) -> Void = { command in
            _ = BrainBarLifecycleWatchdog.run(command: command)
        }
    ) {
        self.configuration = configuration
        self.processProvider = processProvider
        self.terminateProcess = terminateProcess
        self.relaunch = relaunch
    }

    public func start() {
        queue.async { [weak self] in
            guard let self, self.timer == nil else { return }
            let timer = DispatchSource.makeTimerSource(queue: self.queue)
            timer.schedule(
                deadline: .now() + self.configuration.checkInterval,
                repeating: self.configuration.checkInterval
            )
            timer.setEventHandler { [weak self] in
                self?.check()
            }
            self.timer = timer
            timer.resume()
        }
    }

    public func stop() {
        queue.sync {
            timer?.cancel()
            timer = nil
            isRestarting = false
        }
    }

    private func check() {
        guard !isRestarting else { return }
        guard Self.isHeartbeatStale(
            atPath: configuration.heartbeatPath,
            now: Date(),
            timeout: configuration.staleTimeout
        ) else {
            return
        }

        let pids = processProvider()
        guard !pids.isEmpty else {
            isRestarting = true
            NSLog("[BrainBarWatchdog] %@ heartbeat is stale and no process is running; requesting launch", configuration.watchedName)
            relaunch(configuration.relaunchCommand)
            queue.asyncAfter(deadline: .now() + configuration.terminateGraceInterval) { [weak self] in
                self?.isRestarting = false
            }
            return
        }

        isRestarting = true
        NSLog(
            "[BrainBarWatchdog] %@ heartbeat stale for %.0fs; terminating PIDs %@",
            configuration.watchedName,
            configuration.staleTimeout,
            pids.map(String.init).joined(separator: ",")
        )
        pids.forEach { terminateProcess($0, SIGTERM) }
        let stalePIDs = pids
        queue.asyncAfter(deadline: .now() + configuration.terminateGraceInterval) { [weak self] in
            guard let self else { return }
            stalePIDs.forEach { self.terminateProcess($0, SIGKILL) }
            self.relaunch(self.configuration.relaunchCommand)
            self.isRestarting = false
        }
    }

    public static func isHeartbeatStale(atPath path: String, now: Date, timeout: TimeInterval) -> Bool {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
              let modifiedAt = attributes[.modificationDate] as? Date
        else {
            return true
        }
        return now.timeIntervalSince(modifiedAt) > timeout
    }

    public static func writeHeartbeat(to path: String) {
        let url = URL(fileURLWithPath: path)
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let payload = "\(Date().timeIntervalSince1970)\n"
        try? payload.write(to: url, atomically: true, encoding: .utf8)
    }

    public static func makeHeartbeatTimer(path: String, interval: TimeInterval, queue: DispatchQueue) -> DispatchSourceTimer {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now(), repeating: interval)
        timer.setEventHandler {
            writeHeartbeat(to: path)
        }
        timer.resume()
        return timer
    }

    public static func runningPIDs(named executableName: String, bundleIdentifiers: [String] = []) -> [pid_t] {
        let currentPID = ProcessInfo.processInfo.processIdentifier
        let appPIDs = NSWorkspace.shared.runningApplications.compactMap { app -> pid_t? in
            guard app.processIdentifier != currentPID else { return nil }
            if bundleIdentifiers.contains(app.bundleIdentifier ?? "") {
                return app.processIdentifier
            }
            if app.localizedName == executableName || app.executableURL?.lastPathComponent == executableName {
                return app.processIdentifier
            }
            return nil
        }
        if !appPIDs.isEmpty {
            return Array(Set(appPIDs))
        }

        return pgrep(executableName).filter { $0 != currentPID }
    }

    private static func pgrep(_ executableName: String) -> [pid_t] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
        process.arguments = ["-x", executableName]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
        } catch {
            return []
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0,
              let output = String(data: data, encoding: .utf8)
        else {
            return []
        }
        return output
            .components(separatedBy: .newlines)
            .compactMap { Int32($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            .map { pid_t($0) }
    }

    private static func run(command: RelaunchCommand) -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: command.executablePath)
        process.arguments = command.arguments
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            NSLog("[BrainBarWatchdog] Failed to run %@ %@: %@", command.executablePath, command.arguments.joined(separator: " "), String(describing: error))
            return false
        }
    }

    public static func makeDaemonWatchdog() -> BrainBarLifecycleWatchdog {
        BrainBarLifecycleWatchdog(
            configuration: Configuration(
                watchedName: "BrainBarDaemon",
                heartbeatPath: daemonHeartbeatPath,
                relaunchCommand: .launchctlKickstart(label: daemonLaunchAgentLabel)
            ),
            processProvider: {
                runningPIDs(named: "BrainBarDaemon", bundleIdentifiers: ["com.brainlayer.brainbar-daemon", "com.brainlayer.BrainBarDaemon"])
            }
        )
    }

    public static func makeUIWatchdog(bundlePath: String = Bundle.main.bundlePath) -> BrainBarLifecycleWatchdog {
        BrainBarLifecycleWatchdog(
            configuration: Configuration(
                watchedName: "BrainBar",
                heartbeatPath: uiHeartbeatPath,
                relaunchCommand: .launchctlKickstart(label: uiLaunchAgentLabel)
            ),
            processProvider: {
                runningPIDs(named: "BrainBar", bundleIdentifiers: ["com.brainlayer.BrainBar"])
            },
            relaunch: { command in
                let launchctlSucceeded = run(command: command)
                if !launchctlSucceeded, FileManager.default.fileExists(atPath: bundlePath) {
                    _ = run(command: .openBundle(bundlePath))
                }
            }
        )
    }
}
