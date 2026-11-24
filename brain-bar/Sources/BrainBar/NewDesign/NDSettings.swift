import AppKit
import SwiftUI

/// Settings (⌘,): account, design, settings file, Diagnostics and the meaning-search explanation.
@MainActor final class NDSettingsWindowController: NSObject, NSWindowDelegate {
    static let width: CGFloat = 480
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
        window.backgroundColor = NDColor.bg
        window.contentViewController = hosting
        window.setContentSize(size)
        window.contentMinSize = size
        window.contentMaxSize = size
        window.title = BrainBarMenuPresenter.SurfaceSay.settingsTitle
        // Its own identity: the app's empty SwiftUI Settings scene window ("<app> Settings") is a different window.
        window.identifier = NSUserInterfaceItemIdentifier("brainbar.nd.settings")
        window.isReleasedWhenClosed = false
        window.isRestorable = false
        super.init()
        window.delegate = self
    }

    /// The 28 pt title bar the content runs under, with its hairline.
    static let titleBar: CGFloat = 28
    /// A fallback before the form's own fitting height is available.
    static let contentHeight: CGFloat = 355

    static func contentSize(of view: NSView) -> NSSize {
        view.frame = NSRect(x: 0, y: 0, width: width, height: contentHeight)
        view.layoutSubtreeIfNeeded()
        let fitting = view.fittingSize
        let height = fitting.height.isFinite && fitting.height > 0 ? ceil(fitting.height) : contentHeight
        return NSSize(width: width, height: height)
    }

    /// Opens the one Settings window (the same one on every call), front and key.
    /// `present` injects only presentation for headless tests; preparation is the real OPEN=settings path.
    static func show(store: NDStore, activationPolicy: BrainBarWindowActivationPolicy? = nil,
                     present: ((NDSettingsView) -> Void)? = nil) {
        store.windowActivationPolicy = activationPolicy
        store.refreshSettingsFileSummary()
        if let present {
            present(NDSettingsView(store: store))
            return
        }
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
    var showsUpdateControl = true
    @AppStorage(BrainBarDesign.key) private var design: String = BrainBarDesign.defaultValue.rawValue
    var configPath: String = "~/.config/brainlayer/brainlayer.env"
    private typealias Say = BrainBarMenuPresenter.SurfaceSay

    static func designIndex(for stored: String) -> Int {
        let resolved = BrainBarDesign(rawValue: stored) ?? BrainBarDesign.defaultValue
        return BrainBarDesign.allCases.firstIndex(of: resolved)!
    }

    var body: some View {
        let modelLines = BrainBarSettingsExplanation.model(BrainBarEmbeddingModel.configuredName)
        VStack(spacing: 0) {
            ND.bar.frame(height: NDSettingsWindowController.titleBar)
                .overlay(alignment: .bottom) { ND.hair.frame(height: 0.5) }
            VStack(alignment: .leading, spacing: 0) {
                VStack(spacing: 0) {
                    if showsUpdateControl {
                        BrainBarUpdateRow(updates: store.updates)
                        ND.hair.frame(height: 0.5)
                    }
                    row(Say.design) {
                        Spacer()
                        NDSegments(titles: BrainBarDesign.allCases.map(\.title),
                                   selection: Binding(get: { Self.designIndex(for: design) },
                                                      set: { design = BrainBarDesign.allCases[$0].rawValue }),
                                   accessibilityLabel: Say.design, fontSize: 11, padding: 10, controlSize: .small)
                            .fixedSize()
                    }
                    ND.hair.frame(height: 0.5)
                    row(Say.stageDrive) {
                        Spacer()
                        Text(driveState).font(NDType.body).foregroundStyle(store.model.drive.action != nil ? ND.redText : ND.t2)
                        driveAction
                    }
                    if Self.driveAction(store.r1) == nil {
                        BrainBarDriveRetentionLink(evidence: store.inputs.driveRetention)
                            .padding(.horizontal, 12).padding(.bottom, 10)
                    }
                    ND.hair.frame(height: 0.5)
                    row(Say.settingsFile) {
                        Spacer()
                        NDButton(label: Say.showInFinder, small: true) { store.revealSettingsFile() }
                            .help(store.settingsFileURL?.path ?? configPath)
                    }
                    if let line = store.settingsBackupLine {
                        Text(line).font(NDType.caption).foregroundStyle(ND.t2)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 12).padding(.bottom, 10)
                    }
                    ND.hair.frame(height: 0.5)
                    row(Say.diagnostics) {
                        Spacer()
                        NDButton(label: Say.openDiagnostics, small: true) { store.showDiagnostics() }
                    }
                }
                .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(ND.raised))
                .ndDecoration(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(ND.hair2, lineWidth: 0.5))
                Text(Say.classicCaption).font(NDType.caption).foregroundStyle(ND.t3)
                    .padding(.horizontal, 12).padding(.top, 8)
                if !modelLines.isEmpty {
                    Text(Say.meaningModel).font(NDType.bodyStrong).padding(.top, 16)
                    ForEach(modelLines, id: \.self) { line in
                        Text(line).font(NDType.caption).foregroundStyle(ND.t2)
                            .fixedSize(horizontal: false, vertical: true).padding(.top, 4)
                    }
                }
            }
            .padding(20)
        }
        .ignoresSafeArea(.container, edges: .top)
        .frame(width: NDSettingsWindowController.width)
        .background(ND.bg)
        .foregroundStyle(ND.t1)
        .environment(\.ndPalette, .window)
    }

    /// "Signed in" / "Signed out" (or "—" when the last check cannot say), from the accepted Backups row (F01).
    private var driveState: String {
        let state = store.model.drive.state
        if Self.driveAction(store.r1) != nil { return Say.signedOutValue }
        return state == NDModel.Number.unknown.value ? state : Say.signedIn
    }

    private func row<Content: View>(_ label: String, @ViewBuilder _ content: () -> Content) -> some View {
        HStack(spacing: 12) {
            Text(label).font(NDType.body).foregroundStyle(ND.t1).frame(width: 120, alignment: .leading)
            content()
        }
        .padding(.horizontal, 12)
        .frame(height: NDGrid.formRow)
    }

    /// L8: no Disconnect. R1.1 change 26: no Reconnect while signed in; signed out or expired, Reconnect is the window's
    /// one filled button; one click, then the waiting line.
    /// R1.10: Settings shows the same action as Google Drive's station panel and its problem row, with the same words
    /// ("Reconnect"), whenever Drive is signed out, and nothing when it is not: the Drive problem's own action.
    static func driveAction(_ r1: NDR1) -> BrainBarMenuPresentation.Action? {
        r1.items.first { $0.id == "drive" && $0.severity == .red }?.action
    }

    @ViewBuilder private var driveAction: some View {
        if let action = Self.driveAction(store.r1) {
            NDWaitingSwap(waiting: NDDriveWaiting.shows(store.inputs, for: action)) {
                NDButton(label: NDPipelineChart.shortLabel(action), filled: true, small: true, accessibility: action.label) {
                    store.reconnectDrive()
                }
            } line: {
                Text(Say.waitingForBrowser).font(NDType.body).foregroundStyle(ND.t2)
            }
        }
    }
}
