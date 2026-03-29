import SwiftUI

struct StatusPopoverView: View {
    @ObservedObject var collector: StatsCollector

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .center, spacing: 10) {
                Image(systemName: collector.state.symbolName)
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(Color(nsColor: collector.state.color))

                VStack(alignment: .leading, spacing: 2) {
                    Text("BrainBar")
                        .font(.system(size: 15, weight: .semibold))
                    Text(collector.state.label)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(Color(nsColor: collector.state.color))
                }

                Spacer()

                Text(byteString(collector.stats.databaseSizeBytes))
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 8) {
                metricTile("Chunks", value: "\(collector.stats.chunkCount)")
                metricTile("Enriched", value: "\(collector.stats.enrichedChunkCount)")
                metricTile("Pending", value: "\(collector.stats.pendingEnrichmentCount)")
            }

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Recent Activity")
                        .font(.system(size: 12, weight: .semibold))
                    Spacer()
                    Text("\(Int(collector.stats.enrichmentPercent.rounded()))% enriched")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.secondary)
                }

                SparklineShape(values: collector.stats.recentActivityBuckets)
                    .stroke(Color(nsColor: collector.state.color), style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
                    .frame(height: 42)
                    .padding(.vertical, 4)
                    .background(
                        RoundedRectangle(cornerRadius: 10)
                            .fill(Color(nsColor: .windowBackgroundColor))
                    )
            }

            if let daemon = collector.daemon {
                HStack(spacing: 12) {
                    detailLine("PID", "\(daemon.pid)")
                    detailLine("RSS", byteString(Int64(daemon.rssBytes)))
                    detailLine("Sockets", "\(daemon.openConnections)")
                }
            } else {
                Text("Daemon metrics unavailable")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }

            HStack {
                Button("Refresh") {
                    collector.refresh(force: true)
                }
                Button("Quit BrainBar") {
                    NSApplication.shared.terminate(nil)
                }
                Spacer()
            }
        }
        .padding(14)
        .frame(width: 340)
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

    private func detailLine(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
        }
    }

    private func byteString(_ value: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: value, countStyle: .file)
    }
}

private struct SparklineShape: Shape {
    let values: [Int]

    func path(in rect: CGRect) -> Path {
        var path = Path()
        guard values.count > 1 else { return path }

        let maxValue = max(values.max() ?? 0, 1)
        let step = rect.width / CGFloat(max(values.count - 1, 1))

        for (index, value) in values.enumerated() {
            let x = rect.minX + CGFloat(index) * step
            let y = rect.maxY - (CGFloat(value) / CGFloat(maxValue)) * max(rect.height - 2, 1)
            if index == 0 {
                path.move(to: CGPoint(x: x, y: y))
            } else {
                path.addLine(to: CGPoint(x: x, y: y))
            }
        }

        return path
    }
}
