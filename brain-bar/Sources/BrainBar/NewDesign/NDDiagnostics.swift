import AppKit
import SwiftUI

/// One list from the same accepted capture as the per-problem reports; no probes of its own.
struct NDDiagnosticsPresentation {
    struct Row {
        let text: String
        let state: BrainBarHealthVerdict.State
    }
    let rows: [Row]
    var lines: [String] { rows.map(\.text) }

    init(_ inputs: NDInputs) {
        let checks = inputs.capture.verdict?.checks.map { check in
            let label = BrainBarHealthVerdict.Check.labels[check.name] ?? check.name
            let state = BrainBarMenuPresenter.SurfaceSay.diagnosticState(check.state)
            // Check.text's empty-note rule: no separator when there is no note.
            return Row(text: "\(label): \(state)" + (check.note.isEmpty ? "" : " — \(check.note)"), state: check.state)
        } ?? [Row(text: BrainBarMenuPresenter.SurfaceSay.checksUnreadable, state: .unknown)]
        rows = checks + inputs.extras.map { Row(text: $0.text, state: $0.level == .fault ? .red : .amber) }
    }
}

@MainActor final class NDDiagnosticsWindowController: NSObject, NSWindowDelegate {
    private static var current: NDDiagnosticsWindowController?
    private let window: NSWindow
    private let store: NDStore
    private var activationPolicy: BrainBarWindowActivationPolicy?

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

    static func show(store: NDStore, activationPolicy: BrainBarWindowActivationPolicy?) {
        let controller = current ?? NDDiagnosticsWindowController(store: store)
        current = controller
        controller.store.setDiagnosticsActive(true)
        activationPolicy?.windowOpened(controller.window)
        controller.activationPolicy = activationPolicy
        if !controller.window.isVisible { controller.window.center() }
        NSApp.activate(ignoringOtherApps: true)
        controller.window.makeKeyAndOrderFront(nil)
    }

    func windowWillClose(_ notification: Notification) {
        activationPolicy?.windowClosed(window)
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
                    ForEach(Array(NDDiagnosticsPresentation(store.inputs).rows.enumerated()), id: \.offset) { _, row in
                        Text(row.text).font(.system(size: 13)).textSelection(.enabled)
                            .foregroundStyle(row.state == .red ? Color.red : Color.primary)
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
