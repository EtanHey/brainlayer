import Foundation

enum BrainBusEventType: String, Codable, Equatable, Sendable {
    case queueDepth = "queue_depth"
    case enrichStatus = "enrich_status"
    case lastChunkID = "last_chunk_id"
    case dbBusy = "db_busy"
    case healthTick = "health_tick"
}

struct BrainBusEvent: Codable, Equatable, Sendable {
    let type: BrainBusEventType
    let sequence: Int
    let generatedAt: Date
    let queueDepth: Int?
    let enrichStatus: String?
    let lastChunkID: String?
    let dbBusy: Bool?
    let openConnections: Int?

    enum CodingKeys: String, CodingKey {
        case type
        case sequence
        case generatedAt = "generated_at"
        case queueDepth = "queue_depth"
        case enrichStatus = "enrich_status"
        case lastChunkID = "last_chunk_id"
        case dbBusy = "db_busy"
        case openConnections = "open_connections"
    }

    static func queueDepth(_ depth: Int) -> BrainBusEvent {
        BrainBusEvent(type: .queueDepth, queueDepth: depth)
    }

    static func enrichStatus(_ status: String) -> BrainBusEvent {
        BrainBusEvent(type: .enrichStatus, enrichStatus: status)
    }

    static func lastChunkID(_ chunkID: String) -> BrainBusEvent {
        BrainBusEvent(type: .lastChunkID, lastChunkID: chunkID)
    }

    static func dbBusy(_ busy: Bool) -> BrainBusEvent {
        BrainBusEvent(type: .dbBusy, dbBusy: busy)
    }

    static func healthTick(openConnections: Int) -> BrainBusEvent {
        BrainBusEvent(type: .healthTick, openConnections: openConnections)
    }

    func withSequence(_ sequence: Int, generatedAt: Date = Date()) -> BrainBusEvent {
        BrainBusEvent(
            type: type,
            sequence: sequence,
            generatedAt: generatedAt,
            queueDepth: queueDepth,
            enrichStatus: enrichStatus,
            lastChunkID: lastChunkID,
            dbBusy: dbBusy,
            openConnections: openConnections
        )
    }

    private init(
        type: BrainBusEventType,
        sequence: Int = 0,
        generatedAt: Date = Date(),
        queueDepth: Int? = nil,
        enrichStatus: String? = nil,
        lastChunkID: String? = nil,
        dbBusy: Bool? = nil,
        openConnections: Int? = nil
    ) {
        self.type = type
        self.sequence = sequence
        self.generatedAt = generatedAt
        self.queueDepth = queueDepth
        self.enrichStatus = enrichStatus
        self.lastChunkID = lastChunkID
        self.dbBusy = dbBusy
        self.openConnections = openConnections
    }
}

final class BrainBusEventHub: @unchecked Sendable {
    typealias SubscriptionID = UUID
    typealias EventWriter = @Sendable (BrainBusEvent) -> Bool

    private struct Subscriber {
        let writer: EventWriter
        let drainQueue: DispatchQueue
        var buffer: [BrainBusEvent] = []
        var isDraining = false
    }

    private static let queueKey = DispatchSpecificKey<UUID>()
    private let queue = DispatchQueue(label: "com.brainlayer.brainbar.brain-bus")
    private let queueID = UUID()
    private let bufferCapacity: Int
    private var nextSequence = 0
    private var subscribers: [SubscriptionID: Subscriber] = [:]
    private var heartbeatTimer: DispatchSourceTimer?

    init(bufferCapacity: Int = 64, heartbeatInterval: TimeInterval? = 5) {
        self.bufferCapacity = max(1, bufferCapacity)
        queue.setSpecific(key: Self.queueKey, value: queueID)
        if let heartbeatInterval {
            let timer = DispatchSource.makeTimerSource(queue: queue)
            timer.schedule(deadline: .now() + heartbeatInterval, repeating: heartbeatInterval, leeway: .milliseconds(100))
            timer.setEventHandler { [weak self] in
                self?.publishOnQueue(.healthTick(openConnections: self?.subscribers.count ?? 0))
            }
            timer.resume()
            heartbeatTimer = timer
        }
    }

    deinit {
        heartbeatTimer?.cancel()
    }

    func subscribe(_ writer: @escaping EventWriter) -> SubscriptionID {
        let id = UUID()
        queue.sync {
            subscribers[id] = Subscriber(
                writer: writer,
                drainQueue: DispatchQueue(label: "com.brainlayer.brainbar.brain-bus.subscriber.\(id.uuidString)")
            )
        }
        return id
    }

    func unsubscribe(_ id: SubscriptionID) {
        queue.async {
            self.subscribers.removeValue(forKey: id)
        }
    }

    func unsubscribeSynchronously(_ id: SubscriptionID) {
        if DispatchQueue.getSpecific(key: Self.queueKey) == queueID {
            subscribers.removeValue(forKey: id)
            return
        }
        _ = queue.sync {
            self.subscribers.removeValue(forKey: id)
        }
    }

    func publish(_ event: BrainBusEvent) {
        queue.async {
            self.publishOnQueue(event)
        }
    }

    private func publishOnQueue(_ event: BrainBusEvent) {
        nextSequence += 1
        let sequenced = event.withSequence(nextSequence)
        for id in subscribers.keys {
            enqueue(sequenced, for: id)
        }
    }

    private func enqueue(_ event: BrainBusEvent, for id: SubscriptionID) {
        guard var subscriber = subscribers[id] else { return }
        if subscriber.buffer.count >= bufferCapacity {
            subscriber.buffer.removeFirst()
        }
        subscriber.buffer.append(event)
        let shouldStartDrain = !subscriber.isDraining
        if shouldStartDrain {
            subscriber.isDraining = true
        }
        subscribers[id] = subscriber
        if shouldStartDrain {
            drainNext(for: id)
        }
    }

    private func drainNext(for id: SubscriptionID) {
        guard var subscriber = subscribers[id] else { return }
        guard !subscriber.buffer.isEmpty else {
            subscriber.isDraining = false
            subscribers[id] = subscriber
            return
        }
        let event = subscriber.buffer.removeFirst()
        let writer = subscriber.writer
        let drainQueue = subscriber.drainQueue
        subscribers[id] = subscriber

        drainQueue.async { [weak self] in
            let keepOpen = writer(event)
            self?.queue.async { [weak self] in
                guard keepOpen else {
                    self?.subscribers.removeValue(forKey: id)
                    return
                }
                self?.drainNext(for: id)
            }
        }
    }
}
