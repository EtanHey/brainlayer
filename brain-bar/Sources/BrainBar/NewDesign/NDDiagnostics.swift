import AppKit
import SwiftUI

/// One list from the same accepted capture as the per-problem reports; no probes of its own.
struct NDDiagnosticsPresentation {
    let lines: [String]

    init(_ inputs: NDInputs) {
        let checks = inputs.capture.verdict?.checks.map {
            "\(BrainBarHealthVerdict.Check.labels[$0.name] ?? $0.name): \($0.state.rawValue) — \($0.note)"
        } ?? [BrainBarMenuPresenter.SurfaceSay.checksUnreadable]
        lines = checks + inputs.extras.map(\.text)
    }
}

@MainActor final class NDDiagnosticsWindowController: NSObject, NSWindowDelegate {
    private static var current: NDDiagnosticsWindowController?
    private let window: NSWindow
    private let store: NDStore

    private init(store: NDStore) {
        self.store = store
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 560, height: 460),
                          styleMask: [.titled, .closable, .resizable], backing: .buffered, defer: false)
        super.init()
        window.title = BrainBarMenuPresenter.SurfaceSay.diagnostics
        window.contentViewController = NSHostingController(rootView: NDDiagnosticsView(store: store))
        window.isReleasedWhenClosed = false
        window.isRestorable = false
        window.delegate = self
    }

    static func show(store: NDStore) {
        let controller = current ?? NDDiagnosticsWindowController(store: store)
        current = controller
        controller.store.setDiagnosticsActive(true)
        if !controller.window.isVisible { controller.window.center() }
        NSApp.activate(ignoringOtherApps: true)
        controller.window.makeKeyAndOrderFront(nil)
    }

    func windowWillClose(_ notification: Notification) {
        store.setDiagnosticsActive(false)
        Self.current = nil
    }
}

private struct NDDiagnosticsView: View {
    @ObservedObject var store: NDStore
    private typealias Say = BrainBarMenuPresenter.SurfaceSay

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(store.r1.footerLeft).font(.system(size: 11)).foregroundStyle(.secondary)
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(Array(NDDiagnosticsPresentation(store.inputs).lines.enumerated()), id: \.offset) { _, line in
                        Text(line).font(.system(size: 13)).textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            Button(Say.copyFullReport) { store.copyReport() }
            if store.inputs.copyReportFailed {
                Text(Say.copyFailed).font(.system(size: 11))
            }
        }
        .padding(20)
        .frame(minWidth: 400, minHeight: 240)
    }
}
