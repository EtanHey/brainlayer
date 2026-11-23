import AppKit
import XCTest
@testable import BrainBar

/// L57 proofs use offscreen page measurements and pure geometry; no NSWindow is created.
@MainActor
final class BrainBarGapL57Tests: XCTestCase {
    private let titleBar: CGFloat = 52

    func testTwoFaultsOverviewGrowsOnTallScreenAndCalmShrinksKeepingTopLeft() {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm))
        let screen = NSRect(x: 120, y: 80, width: 1400, height: 1200)
        let calm = NDPageHeight.window(store, page: .overview, screen: screen.height, titleBarHeight: titleBar)
        let initial = NSRect(x: 200, y: screen.maxY - calm - titleBar, width: 900, height: calm + titleBar)
        store.setInputsForTesting(BrainBarDesignFixture.inputs(.twoFaults))
        let content = NDPageHeight.content(store, page: .overview) + NDGrid.pageTop + 20
        XCTAssertEqual(content, 809.5, accuracy: 1, "L63 removes 20 pt of Pipeline height and restores the L54 gap; L57 follows the content")
        let height = NDPageHeight.window(store, page: .overview, screen: screen.height, titleBarHeight: titleBar)
        XCTAssertGreaterThan(height, calm)
        XCTAssertGreaterThanOrEqual(height - NDGrid.titleBar - NDGrid.footer, content, "no scroll below the screen cap")
        let grown = NDPageHeight.frame(size: NSSize(width: 900, height: height + titleBar), keeping: initial, visibleFrame: screen)
        XCTAssertEqual(grown.minX, initial.minX)
        XCTAssertEqual(grown.maxY, initial.maxY)
        XCTAssertTrue(screen.contains(grown))
        store.setInputsForTesting(BrainBarDesignFixture.inputs(.calm))
        let restored = NDPageHeight.window(store, page: .overview, screen: screen.height, titleBarHeight: titleBar)
        XCTAssertEqual(restored, calm)
        XCTAssertEqual(NDPageHeight.frame(size: NSSize(width: 900, height: restored + titleBar), keeping: grown, visibleFrame: screen), initial)
    }

    func testShortVisibleFrameCapsTwoFaultsOverviewAndAllowsScroll() {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.twoFaults))
        let screen = NSRect(x: -1400, y: 30, width: 1400, height: 600)
        let height = NDPageHeight.window(store, page: .overview, screen: screen.height, titleBarHeight: titleBar)
        XCTAssertEqual(height, screen.height - titleBar)
        let content = NDPageHeight.content(store, page: .overview) + NDGrid.pageTop + 20
        XCTAssertGreaterThan(content, height - NDGrid.titleBar - NDGrid.footer, "overflow scrolls above the footer")
        let frame = NDPageHeight.frame(size: NSSize(width: 900, height: height + titleBar),
                                      keeping: NSRect(x: -1300, y: 200, width: 900, height: 400), visibleFrame: screen)
        XCTAssertEqual(frame.width, 900)
        XCTAssertEqual(frame.height, screen.height)
        XCTAssertTrue(screen.contains(frame))
    }

    func testContentPublicationsCoalesceAndRefitEveryPageAfterTheStoreUpdates() async {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm))
        var pages: [NDPage] = []
        var measured: [CGFloat] = []
        let observer = NDPageFitObserver(store: store) {
            pages.append(store.page)
            measured.append(NDPageHeight.content(store, page: store.page))
        }
        for page in NDPage.allCases {
            let count = pages.count
            store.page = page
            store.setInputsForTesting(BrainBarDesignFixture.inputs(.twoFaults))
            observer.schedule()
            observer.schedule()
            // A page and its inputs/model publish separately, but belong to one committed update.
            for _ in 0..<10 { await Task.yield() }
            XCTAssertEqual(pages.count, count + 1)
            XCTAssertEqual(pages.last, page)
            XCTAssertEqual(measured.last, NDPageHeight.content(store, page: page))
        }
        // An input-only mutation also refits (no model or headline change).
        let count = pages.count
        store.reconnectDrive()
        for _ in 0..<10 { await Task.yield() }
        XCTAssertEqual(pages.count, count + 1)
        withExtendedLifetime(observer) {}
    }

    func testEveryPageRefitsItsChangedContentAgainstVisibleFrame() {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm))
        for page in NDPage.allCases {
            store.page = page
            for fixture: BrainBarDesignFixture.Name in [.calm, .twoFaults, .backupsFailedUpload] {
                store.setInputsForTesting(BrainBarDesignFixture.inputs(fixture))
                let needed = ceil(NDPageHeight.content(store, page: page) + NDPageHeight.chrome)
                XCTAssertEqual(NDPageHeight.window(store, page: page, screen: 2000, titleBarHeight: titleBar), needed)
                XCTAssertEqual(NDPageHeight.window(store, page: page, screen: 500, titleBarHeight: titleBar), min(needed, 448))
            }
        }
    }
}
