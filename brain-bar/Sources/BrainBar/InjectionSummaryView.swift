import SwiftUI

struct InjectionSummaryView: View {
    let events: [InjectionEvent]

    private var totalTokens: Int {
        events.reduce(0) { $0 + $1.tokenCount }
    }

    private var totalChunks: Int {
        events.reduce(0) { $0 + $1.chunkCount }
    }

    var body: some View {
        HStack(spacing: 8) {
            metricTile("Injects", value: "\(events.count)")
            metricTile("Chunks", value: "\(totalChunks)")
            metricTile("Tokens", value: "\(totalTokens)")
        }
    }

    private func metricTile(_ label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 18, weight: .semibold, design: .rounded))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
    }
}
