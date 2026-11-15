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
        // The Design control writes the store's preferences: the app's own, or a capture fixture's separate suite (F06).
        let hosting = NSHostingController(rootView: NDSettingsView(store: store).defaultAppStorage(store.preferences))
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

    /// Switching to Classic closes the new design's Settings window and lets it go (N02).
    static func closeIfOpen() {
        current?.window.close()
        current = nil
    }

    static var isOpenForTesting: Bool { current != nil }

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
                Text(store.model.drive.state).font(NDType.row).foregroundStyle(ND.ink2)
                Spacer()
                driveAction
            }
            ND.rule.frame(height: 1)
            row("Design") {
                Spacer()
                NDSegments(titles: BrainBarDesign.allCases.map(\.title),
                           selection: Binding(get: { BrainBarDesign.allCases.firstIndex { $0.rawValue == design } ?? 0 },
                                              set: { design = BrainBarDesign.allCases[$0].rawValue }),
                           accessibilityLabel: "Design", fontSize: 12, padding: 12)
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

    /// F01: the state and the action come from the same accepted Backups row and counted fault every other surface
    /// draws (a stale or duplicated check reads "—"). L8: no Disconnect in this version; Reconnect and Connect are one click.
    @ViewBuilder private var driveAction: some View {
        if let action = store.model.drive.action {
            if store.inputs.driveReconnecting {
                HStack(spacing: 7) {
                    NDMarkView(mark: .running)
                    Text(BrainBarMenuPresenter.SurfaceSay.waitingForBrowser).font(NDType.fact).foregroundStyle(ND.ink2)
                }
            } else {
                Button(action.kind == .connectDrive ? "Connect" : "Reconnect") { store.reconnectDrive() }
                    .buttonStyle(NDLinkButtonStyle())
            }
        }
    }
}
