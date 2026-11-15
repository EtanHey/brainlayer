import AppKit
import SwiftUI

/// The standard Settings window (⌘,), 520 pt wide, three rows (DESIGN-SPEC § 6.5). Job switches live on the Jobs page;
/// nothing here repeats it.
@MainActor final class NDSettingsWindowController: NSObject, NSWindowDelegate {
    static let width: CGFloat = 520
    private static var current: NDSettingsWindowController?

    let window: NSWindow
    let store: NDStore

    init(store: NDStore) {
        self.store = store
        let hosting = NSHostingController(rootView: NDSettingsView(store: store))
        hosting.sizingOptions = [.preferredContentSize]
        window = NSWindow(contentViewController: hosting)
        window.title = "BrainBar Settings"
        window.styleMask = [.titled, .closable]
        window.isReleasedWhenClosed = false
        window.isRestorable = false
        super.init()
        window.delegate = self
    }

    /// Opens the one Settings window (the same one on every call), front and key.
    static func show(store: NDStore, activationPolicy: BrainBarWindowActivationPolicy? = nil) {
        let controller = current ?? NDSettingsWindowController(store: store)
        current = controller
        if !controller.window.isVisible { controller.window.center() }
        activationPolicy?.windowOpened(controller.window)
        controller.activationPolicy = activationPolicy
        NSApp.activate(ignoringOtherApps: true)
        controller.window.makeKeyAndOrderFront(nil)
    }

    private var activationPolicy: BrainBarWindowActivationPolicy?

    func windowWillClose(_ notification: Notification) {
        activationPolicy?.windowClosed(window)
    }
}

struct NDSettingsView: View {
    @ObservedObject var store: NDStore
    @AppStorage(BrainBarDesign.key) private var design: String = BrainBarDesign.defaultValue.rawValue
    var configPath: String = "~/.config/brainlayer/brainlayer.env"

    var body: some View {
        VStack(spacing: 0) {
            row("Google Drive") {
                Text(driveState).font(NDType.row).foregroundStyle(ND.ink2)
                Spacer()
                driveAction
            }
            ND.rule.frame(height: 1)
            row("Design") {
                Spacer()
                Picker("Design", selection: $design) {
                    ForEach(BrainBarDesign.allCases, id: \.rawValue) { Text($0.title).tag($0.rawValue) }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .fixedSize()
            }
            ND.rule.frame(height: 1)
            row("Settings file") {
                Spacer()
                Button("Show in Finder") { store.revealSettingsFile() }
                    .buttonStyle(NDLinkButtonStyle())
                    .help(configPath)
            }
        }
        .padding(EdgeInsets(top: 14, leading: 24, bottom: 18, trailing: 24))
        .frame(width: NDSettingsWindowController.width)
        .background(ND.night)
        .foregroundStyle(ND.ink)
        .environment(\.ndPalette, .window)
    }

    private func row<Content: View>(_ group: String, @ViewBuilder _ content: () -> Content) -> some View {
        HStack(spacing: 0) {
            Text(group).font(NDType.settingsGroup).tracking(0.2).frame(width: 132, alignment: .leading)
                .accessibilityAddTraits(.isHeader)
            content()
        }
        .frame(minHeight: 40)
    }

    private var drive: BrainBarHealthVerdict.State? {
        store.inputs.capture.verdict?.checks.first { $0.name == "drive_auth" }?.state
    }

    private var driveState: String {
        switch drive {
        case .green?, .amber?: "Connected"
        case .red?: store.inputs.capture.driveAuthModel == .missing ? "Not connected" : "Disconnected"
        default: "—"
        }
    }

    /// Reconnect and Connect are one click. Disconnect would ask first (N4), but nothing in BrainLayer can disconnect
    /// Drive yet, so no Disconnect is drawn: a control without an action is not drawn (L5).
    @ViewBuilder private var driveAction: some View {
        if drive == .red {
            if store.inputs.driveReconnecting {
                HStack(spacing: 7) {
                    NDMarkView(mark: .running)
                    Text("Waiting for you to approve in your browser…").font(NDType.fact).foregroundStyle(ND.ink2)
                }
            } else {
                let missing = store.inputs.capture.driveAuthModel == .missing
                Button(missing ? "Connect" : "Reconnect") { store.reconnectDrive() }.buttonStyle(NDLinkButtonStyle())
            }
        } else if let disconnect = store.disconnectDrive {
            Button("Disconnect") { NDDisconnectSheet.confirm(perform: disconnect) }.buttonStyle(NDLinkButtonStyle())
        }
    }
}

/// N4: "Disconnect Google Drive?" as a sheet on the key window; Cancel is the default (Return and Escape).
@MainActor enum NDDisconnectSheet {
    static func alert() -> NSAlert {
        let alert = NSAlert()
        alert.messageText = "Disconnect Google Drive?"
        alert.informativeText = "Backups stop uploading until you connect it again."
        let cancel = alert.addButton(withTitle: "Cancel")
        cancel.keyEquivalent = "\r"
        let disconnect = alert.addButton(withTitle: "Disconnect")
        disconnect.hasDestructiveAction = true
        disconnect.keyEquivalent = ""
        return alert
    }

    static func confirm(perform: @escaping () -> Void) {
        let alert = alert()
        guard let window = NSApp.keyWindow else { if alert.runModal() == .alertSecondButtonReturn { perform() }; return }
        alert.beginSheetModal(for: window) { response in if response == .alertSecondButtonReturn { perform() } }
    }
}
