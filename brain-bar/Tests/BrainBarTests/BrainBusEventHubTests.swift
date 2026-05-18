import XCTest
@testable import BrainBar

final class BrainBusEventHubTests: XCTestCase {
    func testSubscribeReceivesThreeStateChangesInOrderWithinLatencyBudget() throws {
        let hub = BrainBusEventHub(bufferCapacity: 8, heartbeatInterval: nil)
        let received = LockedBrainBusEvents()
        let allReceived = XCTestExpectation(description: "received three brain bus events")

        let subscription = hub.subscribe { event in
            received.append(event)
            if received.count == 3 {
                allReceived.fulfill()
            }
            return true
        }
        defer { hub.unsubscribe(subscription) }

        let startedAt = DispatchTime.now()
        hub.publish(.queueDepth(2))
        hub.publish(.enrichStatus("running"))
        hub.publish(.lastChunkID("chunk-3"))

        wait(for: [allReceived], timeout: 1.0)
        let elapsedMillis = Double(DispatchTime.now().uptimeNanoseconds - startedAt.uptimeNanoseconds) / 1_000_000

        XCTAssertLessThan(elapsedMillis, 50)
        XCTAssertEqual(received.snapshot().map(\.type), [.queueDepth, .enrichStatus, .lastChunkID])
    }

    func testBackpressureDropsOldestEventsWithoutStallingPublisher() throws {
        let hub = BrainBusEventHub(bufferCapacity: 3, heartbeatInterval: nil)
        let received = LockedBrainBusEvents()
        let sawLatest = XCTestExpectation(description: "slow subscriber eventually receives latest event")

        let subscription = hub.subscribe { event in
            Thread.sleep(forTimeInterval: 0.02)
            received.append(event)
            if event.sequence == 20 {
                sawLatest.fulfill()
            }
            return true
        }
        defer { hub.unsubscribe(subscription) }

        let startedAt = DispatchTime.now()
        for index in 1...20 {
            hub.publish(.healthTick(openConnections: index))
        }
        let publishElapsedMillis = Double(DispatchTime.now().uptimeNanoseconds - startedAt.uptimeNanoseconds) / 1_000_000

        XCTAssertLessThan(publishElapsedMillis, 20)
        wait(for: [sawLatest], timeout: 1.0)

        let delivered = received.snapshot()
        XCTAssertLessThan(delivered.count, 20)
        XCTAssertEqual(delivered.last?.sequence, 20)
    }

    func testSynchronousUnsubscribeRemovesSubscriberBeforeReturn() throws {
        let hub = BrainBusEventHub(bufferCapacity: 8, heartbeatInterval: nil)
        let received = LockedBrainBusEvents()
        let subscription = hub.subscribe { event in
            received.append(event)
            return true
        }

        hub.unsubscribeSynchronously(subscription)
        hub.publish(.queueDepth(1))
        Thread.sleep(forTimeInterval: 0.05)

        XCTAssertEqual(received.count, 0)
    }
}

private final class LockedBrainBusEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [BrainBusEvent] = []

    var count: Int {
        lock.withLock { events.count }
    }

    func append(_ event: BrainBusEvent) {
        lock.withLock {
            events.append(event)
        }
    }

    func snapshot() -> [BrainBusEvent] {
        lock.withLock { events }
    }
}
