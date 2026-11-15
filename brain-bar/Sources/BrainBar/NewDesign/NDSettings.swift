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
        // Review blocker 2: the window opened larger than its three rows (the first capture shows them as a strip at the
        // top of an empty window). Its size is now its content's, measured and fixed before it is ever shown.
        hosting.sizingOptions = []
        let size = Self.contentSize(of: hosting.view)
        window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.titled, .closable, .fullSizeContentView],
                          backing: .buffered, defer: false)
        // Review polish 3: one material. The 28 pt title bar is transparent over the Night background, with the view's
        // 1 pt Rule hairline under it, as the main window draws at 52 pt.
        window.titlebarAppearsTransparent = true
        window.titlebarSeparatorStyle = .none
        window.backgroundColor = NDColor.night
        window.contentViewController = hosting
        window.setContentSize(size)
        window.contentMinSize = size
        window.contentMaxSize = size
        window.title = "BrainBar Settings"
        // Its own identity: the app's empty SwiftUI Settings scene window ("<app> Settings") is a different window.
        window.identifier = NSUserInterfaceItemIdentifier("brainbar.nd.settings")
        window.isReleasedWhenClosed = false
        window.isRestorable = false
        super.init()
        window.delegate = self
    }

    /// The title bar the content runs under (polish 3), and its hairline.
    static let titleBar: CGFloat = 28
    /// 520 pt wide and as tall as its rows under the title bar: 28 pt bar, 1 pt hairline, three 40 pt rows, two 1 pt
    /// rules, 14 and 18 pt insets (183 pt).
    static let contentHeight: CGFloat = 183

    static func contentSize(of view: NSView) -> NSSize {
        view.frame = NSRect(x: 0, y: 0, width: width, height: contentHeight)
        view.layoutSubtreeIfNeeded()
        let fitting = view.fittingSize
        let height = fitting.height > 0 && fitting.height < 400 ? ceil(fitting.height) : contentHeight
        return NSSize(width: width, height: height)
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
            Color.clear.frame(height: NDSettingsWindowController.titleBar)
                .overlay(alignment: .bottom) { ND.rule.frame(height: 1) }
            rows
        }
        .ignoresSafeArea(.container, edges: .top)
        .frame(width: NDSettingsWindowController.width)
        .background(ND.night)
        .foregroundStyle(ND.ink)
        .environment(\.ndPalette, .window)
    }

    private var rows: some View {
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
            NDWaitingSwap(waiting: store.inputs.driveReconnecting) {
                Button(action.kind == .connectDrive ? "Connect" : "Reconnect") { store.reconnectDrive() }
                    .buttonStyle(NDLinkButtonStyle())
            } line: {
                HStack(spacing: 7) {
                    NDMarkView(mark: .running)
                    Text(BrainBarMenuPresenter.SurfaceSay.waitingForBrowser).font(NDType.fact).foregroundStyle(ND.ink2)
                }
            }
        }
    }
}
