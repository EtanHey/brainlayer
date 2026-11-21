#if DEBUG && BRAINBAR_UI
import AppKit

/// Capture mode's driver, for a capture machine that cannot click. Read only with `BRAINBAR_DESIGN_FIXTURE` set; every
/// step goes through the path a click takes (the status menu popped as #1810's notification click pops it, the
/// window's `showPage`, Settings' ⌘, `showSettings`, the store's page and range the controls set, the Reconnect action).
/// No drawing of its own. Each step prints one `capture:` line to stderr so screenshots can be timed.
@MainActor enum NDCaptureDriver {
    static func start(status: BrainBarStatusPopoverController, panel: BrainBarDashboardPanelController, fixturePage: NDPage,
                      environment: [String: String] = ProcessInfo.processInfo.environment) {
        let store = panel.newDesignStore
        let tab = environment["BRAINBAR_DESIGN_TAB"].flatMap(NDPage.init(rawValue:)) ?? fixturePage
        let range = environment["BRAINBAR_DESIGN_RANGE"].flatMap { ["1h": NDRange.hour, "3h": .threeHours, "24h": .day][$0] }
        let open = environment["BRAINBAR_DESIGN_OPEN"]
        let hold = environment["BRAINBAR_DESIGN_HOLD"].flatMap(Double.init)
        store.page = tab
        if let range { store.range = range }
        // Fix 5: a live waveform with no sampler; every N seconds one invented bucket arrives through the sample path.
        if let tick = environment["BRAINBAR_DESIGN_TICK"].flatMap(Double.init), tick > 0 {
            let timer = Timer(timeInterval: tick, repeats: true) { _ in
                MainActor.assumeIsolated { store.tickFixture(); log("tick \(store.fixtureTicks)") }
            }
            RunLoop.main.add(timer, forMode: .common)
        }
        guard let open else { return panel.showPage(tab) }  // no switch: capture mode as before
        after(1) {
            switch open {
            case "window": panel.showPage(tab); log("opened window \(tab.rawValue)")
            case "settings": panel.showSettings(); log("opened settings")
            case "dropdown":
                if let hold { after(hold) { status.contextMenuForTesting.cancelTracking(); log("closed dropdown") } }
                script(environment["BRAINBAR_DESIGN_SCRIPT"], store: store)
                log("opened dropdown")
                status.openNotificationDestination()  // returns when the menu closes
                return
            default: log("unknown BRAINBAR_DESIGN_OPEN \(open)"); return
            }
            script(environment["BRAINBAR_DESIGN_SCRIPT"], store: store)
        }
    }

    /// Timed sequences through the controls' own state and actions. `menu-keys` is not offered: AppKit has no public way
    /// to move a menu's highlight from inside its tracking loop.
    static func script(_ name: String?, store: NDStore) {
        switch name {
        case "tabs":
            for (step, page) in [NDPage.overview, .backups, .jobs, .overview].enumerated() {
                after(Double(step) * 2) { store.page = page; log("tab \(page.rawValue)") }
            }
        case "ranges":
            for (step, range) in [NDRange.hour, .threeHours, .day, .hour].enumerated() {
                after(Double(step) * 2) { store.range = range; log("range \(range.label)") }
            }
        case "reconnect":
            after(2) { store.reconnectDrive(); log("pressed reconnect") }
        case nil: break
        default: log("unknown BRAINBAR_DESIGN_SCRIPT \(name ?? "")")
        }
    }

    /// A timer in the common modes, so it also fires while a popped menu runs its tracking loop.
    static func after(_ seconds: Double, _ body: @escaping @MainActor () -> Void) {
        let timer = Timer(timeInterval: seconds, repeats: false) { _ in MainActor.assumeIsolated { body() } }
        RunLoop.main.add(timer, forMode: .common)
    }

    static func log(_ line: String) {
        FileHandle.standardError.write(Data(("capture: " + line + "\n").utf8))
    }
}
#endif
