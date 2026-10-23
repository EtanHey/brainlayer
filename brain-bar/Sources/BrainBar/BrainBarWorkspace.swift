import AppKit
import Foundation

/// The Finder and pasteboard side effects, injected so tests never touch either.
protocol BrainBarWorkspaceActing: Sendable {
    func reveal(_ url: URL)
    func copy(_ text: String)
    /// Opens a file in its default app (a log opens in Console).
    func open(_ url: URL)
    /// Log actions need an acknowledged open for files and fallback directories.
    func openLog(_ url: URL) -> Bool
}

extension BrainBarWorkspaceActing {
    func open(_ url: URL) { reveal(url) }
    /// An adapter without an acknowledged opener cannot report success.
    func openLog(_ url: URL) -> Bool { false }
}

struct BrainBarWorkspace: BrainBarWorkspaceActing {
    init() { Self.requireProductionProcess() }

    private static func requireProductionProcess() {
        precondition(
            ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] == nil
                && NSClassFromString("XCTestCase") == nil,
            "Inject BrainBarWorkspaceActing in tests; real workspace access is forbidden"
        )
    }

    func reveal(_ url: URL) {
        Self.requireProductionProcess()
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    func open(_ url: URL) {
        Self.requireProductionProcess()
        NSWorkspace.shared.open(url)
    }

    func openLog(_ url: URL) -> Bool {
        Self.requireProductionProcess()
        return NSWorkspace.shared.open(url)
    }

    func copy(_ text: String) {
        Self.requireProductionProcess()
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

#if DEBUG
/// Recording workspace for isolated render harnesses and XCTest fixtures.
final class BrainBarRecordingWorkspace: BrainBarWorkspaceActing, @unchecked Sendable {
    private let lock = NSLock()
    private var actions: [(String, String)] = []
    var recorded: [(String, String)] { lock.lock(); defer { lock.unlock() }; return actions }
    private func record(_ action: String, _ value: String) {
        lock.lock(); defer { lock.unlock() }; actions.append((action, value))
    }
    func reveal(_ url: URL) { record("reveal", url.path) }
    func open(_ url: URL) { record("open", url.path) }
    func openLog(_ url: URL) -> Bool { open(url); return true }
    func copy(_ text: String) { record("copy", text) }
}
#endif
