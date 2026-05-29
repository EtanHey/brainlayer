import AppKit
import BrainBarLifecycle
import Foundation

@MainActor
enum BrainBarProcessControl {
    static func quit() {
        NSApplication.shared.terminate(nil)
    }

    static func restart(bundlePath: String = Bundle.main.bundlePath) {
        guard FileManager.default.fileExists(atPath: bundlePath) else {
            NSLog("[BrainBar] Cannot restart: bundle path does not exist: %@", bundlePath)
            return
        }

        BrainBarRestartHandoff.markRestartingProcess()

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        process.arguments = ["-n", bundlePath]

        do {
            try process.run()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                NSApplication.shared.terminate(nil)
            }
        } catch {
            BrainBarRestartHandoff.clear()
            NSLog("[BrainBar] Failed to schedule restart: %@", String(describing: error))
        }
    }
}
