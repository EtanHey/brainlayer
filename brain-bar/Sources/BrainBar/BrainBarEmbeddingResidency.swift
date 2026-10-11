import Darwin
import Foundation

/// The embedding model BrainLayer loads. There is no config or env key for it: the hotlane
/// calls `get_embedding_model()` with no argument, so `DEFAULT_MODEL` in
/// `src/brainlayer/embeddings.py` is the only source. `BrainBarEmbeddingResidencyTests`
/// reads that file and fails if the two drift.
enum BrainBarEmbeddingModel {
    static let configuredName = "BAAI/bge-large-en-v1.5"
}

/// What BrainBar can measure about the hotlane, the long-lived process that embeds new
/// memories. The model loads inside it lazily, on the first embed, so a running hotlane is
/// not proof the model is resident — the presentation names the process, never the model.
enum BrainBarEmbeddingProcessState: Equatable, Sendable {
    case running(pid: pid_t, residentBytes: UInt64?)
    case stopped
    case unmeasurable
}

protocol BrainBarEmbeddingResidencySampling: Sendable {
    func sample() -> BrainBarEmbeddingProcessState
}

struct HotlaneEmbeddingResidencyProbe: BrainBarEmbeddingResidencySampling {
    static let launchdLabel = BrainLayerLaunchdJob.hotlane.launchdLabel

    private let processProbe: any WatcherProcessProbing
    private let residentBytes: @Sendable (pid_t) -> UInt64?

    init(
        processProbe: any WatcherProcessProbing = LaunchctlWatcherProcessProbe(
            label: HotlaneEmbeddingResidencyProbe.launchdLabel
        ),
        residentBytes: @escaping @Sendable (pid_t) -> UInt64? = { HotlaneEmbeddingResidencyProbe.residentBytes(of: $0) }
    ) {
        self.processProbe = processProbe
        self.residentBytes = residentBytes
    }

    /// The PID comes from launchctl on every call and is never cached (#972): a restarted
    /// hotlane has a new PID, and a remembered one can name an unrelated process.
    func sample() -> BrainBarEmbeddingProcessState {
        switch processProbe.sample() {
        case let .running(pid):
            .running(pid: pid, residentBytes: residentBytes(pid))
        case .absent:
            .stopped
        case .failure:
            .unmeasurable
        }
    }

    static func residentBytes(of pid: pid_t) -> UInt64? {
        var info = proc_taskinfo()
        let size = Int32(MemoryLayout<proc_taskinfo>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            proc_pidinfo(pid, PROC_PIDTASKINFO, 0, pointer, size)
        }
        guard result == size, info.pti_resident_size > 0 else { return nil }
        return info.pti_resident_size
    }
}

struct StaticBrainBarEmbeddingResidencyProbe: BrainBarEmbeddingResidencySampling {
    let state: BrainBarEmbeddingProcessState

    func sample() -> BrainBarEmbeddingProcessState { state }
}

struct BrainBarModelResidencyPresentation: Equatable {
    struct Row: Equatable, Identifiable {
        let label: String
        let value: String
        var id: String { label }
    }

    let rows: [Row]

    // A value BrainBar cannot measure hides its row; it never renders as "unavailable".
    init(modelName: String, process: BrainBarEmbeddingProcessState) {
        var rows = [Row(label: "Model", value: modelName)]
        switch process {
        case let .running(pid, residentBytes):
            rows.append(Row(label: "Status", value: "Hotlane running · PID \(pid)"))
            if let residentBytes {
                // RSS of the whole hotlane process, which also runs enrichment.
                let memory = ByteCountFormatter.string(
                    fromByteCount: Int64(clamping: residentBytes),
                    countStyle: .memory
                )
                rows.append(Row(label: "Resident memory", value: "Hotlane process · \(memory)"))
            }
        case .stopped:
            rows.append(Row(label: "Status", value: "Hotlane stopped"))
        case .unmeasurable:
            break
        }
        self.rows = rows
    }

    func value(for label: String) -> String? {
        rows.first { $0.label == label }?.value
    }
}
