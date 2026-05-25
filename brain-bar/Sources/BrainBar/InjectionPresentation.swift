import Foundation

struct InjectionPresentation {
    struct Snapshot: Equatable {
        let filteredEvents: [InjectionEvent]
        let windowEvents: [InjectionEvent]
        let summary: Summary
        let bursts: [Burst]
        let burstSections: [BurstSection]
        let ribbonBuckets: [Int]
        let sessions: [SessionSummary]
    }

    struct Summary: Equatable {
        let queryCount: Int
        let chunkCount: Int
        let tokenCount: Int
        let activeSessionCount: Int
    }

    struct Burst: Equatable, Identifiable {
        let sessionID: String
        let topicOrSource: String
        let startDate: Date
        let endDate: Date
        let events: [InjectionEvent]

        var id: String {
            "\(groupKey)|\(endDate.timeIntervalSince1970)"
        }

        var groupKey: String { "\(sessionID)\u{001F}\(topicOrSource)" }
        var queryCount: Int { events.count }
        var chunkCount: Int { events.reduce(0) { $0 + $1.chunkCount } }
        var tokenCount: Int { events.reduce(0) { $0 + $1.tokenCount } }

        var summaryTitle: String {
            let chunkNoun = chunkCount == 1 ? "chunk" : "chunks"
            return "\(chunkCount) \(chunkNoun) injected from \"\(topicOrSource)\""
        }

        var chunkPreviewIDs: [String] {
            Array(events.flatMap(\.chunkIDs).prefix(2))
        }

        func remainingChunkCount(after visibleCount: Int) -> Int {
            max(chunkCount - visibleCount, 0)
        }
    }

    struct BurstSection: Equatable, Identifiable {
        let bucket: RibbonBucket
        let bursts: [Burst]

        var id: String { bucket.rawValue }
    }

    enum RibbonBucket: String, Equatable {
        case lastSixtyMinutes
        case oneToTwoHoursAgo
        case twoToSixHoursAgo
        case todayEarlier
        case yesterday
        case older

        var title: String {
            switch self {
            case .lastSixtyMinutes:
                return "Last 60 min"
            case .oneToTwoHoursAgo:
                return "1-2h ago"
            case .twoToSixHoursAgo:
                return "2-6h ago"
            case .todayEarlier:
                return "Today earlier"
            case .yesterday:
                return "Yesterday"
            case .older:
                return "Older"
            }
        }
    }

    struct SessionSummary: Equatable, Identifiable {
        let sessionID: String
        let queryCount: Int
        let chunkCount: Int
        let tokenCount: Int
        let latestDate: Date

        var id: String { sessionID }
    }

    static func snapshot(
        events: [InjectionEvent],
        filterText: String,
        now: Date,
        windowMinutes: Int = 60,
        burstGapMinutes: Int = 60,
        bucketCount: Int = 12
    ) -> Snapshot {
        let filteredEvents = filter(events: events, needle: filterText)
        let parsedEvents = filteredEvents.compactMap { event in
            parseDate(event.timestamp).map { ParsedEvent(event: event, date: $0) }
        }
        .sorted { $0.date > $1.date }

        let windowStart = now.addingTimeInterval(-Double(windowMinutes) * 60.0)
        let windowEvents = parsedEvents.filter { $0.date > windowStart && $0.date <= now }
        let burstEvents = parsedEvents.filter { $0.date <= now }

        let summary = Summary(
            queryCount: windowEvents.count,
            chunkCount: windowEvents.reduce(0) { $0 + $1.event.chunkCount },
            tokenCount: windowEvents.reduce(0) { $0 + $1.event.tokenCount },
            activeSessionCount: Set(windowEvents.map(\.event.sessionID)).count
        )
        let bursts = makeBursts(from: burstEvents, burstGapMinutes: burstGapMinutes)

        return Snapshot(
            filteredEvents: filteredEvents,
            windowEvents: windowEvents.map(\.event),
            summary: summary,
            bursts: bursts,
            burstSections: makeBurstSections(from: bursts, now: now),
            ribbonBuckets: makeRibbonBuckets(
                from: windowEvents,
                windowStart: windowStart,
                now: now,
                bucketCount: max(bucketCount, 1)
            ),
            sessions: makeSessions(from: windowEvents)
        )
    }

    static func parseDate(_ timestamp: String) -> Date? {
        if let date = fractionalISOFormatter().date(from: timestamp) {
            return date
        }
        if let date = plainISOFormatter().date(from: timestamp) {
            return date
        }
        if let date = sqliteTimestampFormatter().date(from: timestamp) {
            return date
        }
        return nil
    }

    static func shortTime(_ timestamp: String) -> String {
        guard let date = parseDate(timestamp) else {
            return String(timestamp.prefix(16))
        }
        return shortTimeFormatter().string(from: date)
    }

    static func burstRangeText(start: Date, end: Date) -> String {
        if Calendar.current.isDate(start, equalTo: end, toGranularity: .minute) {
            return shortTimeFormatter().string(from: end)
        }
        let formatter = shortTimeFormatter()
        return "\(formatter.string(from: start)) -> \(formatter.string(from: end))"
    }

    private struct ParsedEvent {
        let event: InjectionEvent
        let date: Date
    }

    private static func filter(events: [InjectionEvent], needle: String) -> [InjectionEvent] {
        let trimmed = needle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return events }
        let lowered = trimmed.lowercased()
        return events.filter { event in
            event.sessionID.lowercased().contains(lowered) ||
                event.query.lowercased().contains(lowered) ||
                event.chunkIDs.joined(separator: " ").lowercased().contains(lowered)
        }
    }

    private static func makeBursts(from events: [ParsedEvent], burstGapMinutes: Int) -> [Burst] {
        guard let first = events.first else { return [] }

        let maxGap = Double(max(burstGapMinutes, 1)) * 60.0
        var bursts: [Burst] = []
        var currentSessionID = first.event.sessionID
        var currentTopicOrSource = topicOrSource(for: first.event)
        var currentEvents = [first]
        var newest = first.date
        var oldest = first.date

        func flush() {
            bursts.append(
                Burst(
                    sessionID: currentSessionID,
                    topicOrSource: currentTopicOrSource,
                    startDate: oldest,
                    endDate: newest,
                    events: currentEvents.map(\.event)
                )
            )
        }

        for parsed in events.dropFirst() {
            let topicOrSource = topicOrSource(for: parsed.event)
            let gap = newest.timeIntervalSince(parsed.date)
            if parsed.event.sessionID == currentSessionID,
               topicOrSource == currentTopicOrSource,
               gap < maxGap {
                currentEvents.append(parsed)
                oldest = parsed.date
            } else {
                flush()
                currentSessionID = parsed.event.sessionID
                currentTopicOrSource = topicOrSource
                currentEvents = [parsed]
                newest = parsed.date
                oldest = parsed.date
            }
        }

        flush()
        return bursts
    }

    private static func makeBurstSections(from bursts: [Burst], now: Date) -> [BurstSection] {
        var sections: [BurstSection] = []
        var currentBucket: RibbonBucket?
        var currentBursts: [Burst] = []

        func flush() {
            guard let currentBucket, !currentBursts.isEmpty else { return }
            sections.append(BurstSection(bucket: currentBucket, bursts: currentBursts))
        }

        for burst in bursts {
            let bucket = ribbonBucket(for: burst.endDate, now: now)
            if currentBucket == nil || currentBucket == bucket {
                currentBucket = bucket
                currentBursts.append(burst)
            } else {
                flush()
                currentBucket = bucket
                currentBursts = [burst]
            }
        }

        flush()
        return sections
    }

    static func ribbonBucket(for date: Date, now: Date, calendar: Calendar = .current) -> RibbonBucket {
        let age = now.timeIntervalSince(date)
        if age < 60.0 * 60.0 {
            return .lastSixtyMinutes
        }
        if age < 2.0 * 60.0 * 60.0 {
            return .oneToTwoHoursAgo
        }
        if age < 6.0 * 60.0 * 60.0 {
            return .twoToSixHoursAgo
        }
        if calendar.isDate(date, inSameDayAs: now) {
            return .todayEarlier
        }
        if let yesterday = calendar.date(byAdding: .day, value: -1, to: now),
           calendar.isDate(date, inSameDayAs: yesterday) {
            return .yesterday
        }
        return .older
    }

    private static func topicOrSource(for event: InjectionEvent) -> String {
        let trimmed = event.query.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "retrieval" : trimmed
    }

    private static func makeRibbonBuckets(
        from events: [ParsedEvent],
        windowStart: Date,
        now: Date,
        bucketCount: Int
    ) -> [Int] {
        var buckets = Array(repeating: 0, count: bucketCount)
        let total = now.timeIntervalSince(windowStart)
        guard total > 0 else { return buckets }

        for parsed in events {
            let elapsed = parsed.date.timeIntervalSince(windowStart)
            guard elapsed >= 0 else { continue }
            let ratio = min(max(elapsed / total, 0), 0.999_999)
            let index = min(bucketCount - 1, Int(ratio * Double(bucketCount)))
            buckets[index] += 1
        }
        return buckets
    }

    private static func makeSessions(from events: [ParsedEvent]) -> [SessionSummary] {
        let grouped = Dictionary(grouping: events, by: \.event.sessionID)
        return grouped.map { sessionID, values in
            SessionSummary(
                sessionID: sessionID,
                queryCount: values.count,
                chunkCount: values.reduce(0) { $0 + $1.event.chunkCount },
                tokenCount: values.reduce(0) { $0 + $1.event.tokenCount },
                latestDate: values.map(\.date).max() ?? .distantPast
            )
        }
        .sorted {
            if $0.queryCount == $1.queryCount {
                return $0.latestDate > $1.latestDate
            }
            return $0.queryCount > $1.queryCount
        }
    }

    private static func fractionalISOFormatter() -> ISO8601DateFormatter {
        threadFormatter(
            key: "brainbar.injection.fractional-iso",
            create: {
                let formatter = ISO8601DateFormatter()
                formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
                return formatter
            }
        )
    }

    private static func plainISOFormatter() -> ISO8601DateFormatter {
        threadFormatter(
            key: "brainbar.injection.plain-iso",
            create: {
                let formatter = ISO8601DateFormatter()
                formatter.formatOptions = [.withInternetDateTime]
                return formatter
            }
        )
    }

    private static func sqliteTimestampFormatter() -> DateFormatter {
        threadFormatter(
            key: "brainbar.injection.sqlite-timestamp",
            create: {
                let formatter = DateFormatter()
                formatter.calendar = Calendar(identifier: .gregorian)
                formatter.locale = Locale(identifier: "en_US_POSIX")
                formatter.timeZone = TimeZone(secondsFromGMT: 0)
                formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSSSSS"
                return formatter
            }
        )
    }

    private static func shortTimeFormatter() -> DateFormatter {
        threadFormatter(
            key: "brainbar.injection.short-time",
            create: {
                let formatter = DateFormatter()
                formatter.calendar = Calendar(identifier: .gregorian)
                formatter.locale = Locale(identifier: "en_US_POSIX")
                formatter.timeZone = TimeZone.current
                formatter.dateFormat = "HH:mm"
                return formatter
            }
        )
    }

    private static func threadFormatter<Formatter: NSObject>(
        key: String,
        create: () -> Formatter
    ) -> Formatter {
        let dictionary = Thread.current.threadDictionary
        if let formatter = dictionary[key] as? Formatter {
            return formatter
        }
        let formatter = create()
        dictionary[key] = formatter
        return formatter
    }
}
