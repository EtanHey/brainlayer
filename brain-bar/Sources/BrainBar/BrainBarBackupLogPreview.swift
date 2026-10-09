import Foundation
import SwiftUI

/// UI-only, bounded snapshots. No raw receipt or private URL enters the rendered text.
enum BrainBarBackupLogPreview: Equatable, Sendable {
    case unread, empty, unavailable(String)
    case entries(String, truncated: Bool)

    static let byteLimit = 64 * 1024
    static let entryLimit = 20

    static func read(_ url: URL) -> Self {
        present(BrainBarStableFileReader.readTail(url, maxBytes: byteLimit + 1))
    }

    static func present(_ read: BrainBarStableRead) -> Self {
        switch read {
        case .unreadable: return .unavailable("Log could not be read. It may not exist yet or access may be denied.")
        case .changedWhileReading: return .unavailable("Log changed while reading. Refresh to try again.")
        case let .read(data, _):
            var truncated = data.count > byteLimit
            var text = String(decoding: data.suffix(byteLimit), as: UTF8.self)
            if truncated {
                // The byte boundary can bisect a receipt or secret: discard that entire fragment.
                text = text.firstIndex(of: "\n").map { String(text[text.index(after: $0)...]) } ?? ""
            }
            let lines = text.split(whereSeparator: \.isNewline).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            truncated = truncated || lines.count > entryLimit
            let shown = lines.suffix(entryLimit).map { readableEntry(String($0)) }
            if shown.isEmpty {
                return truncated ? .unavailable("Latest entry exceeds the preview limit. Reveal the log for details.") : .empty
            }
            return .entries(shown.joined(separator: "\n\n"), truncated: truncated)
        }
    }

    private static func readableEntry(_ line: String) -> String {
        guard let row = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any] else {
            return redact(line)
        }
        // A receipt may contain credentials, IDs, paths or nested raw responses. Opt in fields only.
        let keys = [("attempted_at", "Attempted"), ("completed_at", "Completed"), ("timestamp", "Time"),
                    ("status", "Status"), ("stage", "Stage"), ("uploaded", "Uploaded"), ("verified", "Verified"),
                    ("error_type", "Error type"), ("error_code", "Error code"), ("error", "Error"), ("reason", "Reason")]
        let fields = keys.compactMap { key, label -> String? in
            guard let value = row[key], !(value is NSNull) else { return nil }
            if let string = value as? String { return "\(label): \(redact(string))" }
            if let number = value as? NSNumber {
                let shown = CFGetTypeID(number) == CFBooleanGetTypeID() ? (number.boolValue ? "Yes" : "No") : number.stringValue
                return "\(label): \(shown)"
            }
            return nil
        }
        return fields.isEmpty ? "Log entry has no preview fields. Reveal the log for details." : fields.joined(separator: " · ")
    }

    static func redact(_ input: String) -> String {
        let scrubbed = SecretScrubber.scrub(input)
        guard scrubbed.quarantineCount == 0 else {
            let failure = indicatesFailure(input)
            return failure ? "Failure detail hidden for privacy; reveal log for details." : "[private detail hidden; reveal log for details]"
        }
        var text = scrubbed.text
        // Match the whole compound label even when its payload is short or low entropy.
        let credentialLabel = #"(?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]*(?:token|secret|password|authorization|api[_ -]?key)[A-Za-z0-9_.-]*"#
        // URLs (including query credentials), absolute/tilde/Windows paths, credential labels,
        // provider-prefixed and long token-shaped values. Paths consume their remaining segment
        // because unquoted filenames may contain spaces; failure status/type stay separate.
        let rules = [
            // Header values include their scheme and credential, not just the first word.
            (#"(?i)(?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]*authorization[A-Za-z0-9_.-]*[\"']?\s*[:=]\s*(?:[\"'][^\"']*[\"']|[^\r\n,;]+)"#, "[credential hidden]"),
            (#"(?i)\b[a-z][a-z0-9+.-]*://[^\s\"'<>]+"#, "[private URL]"),
            (#"[\"'](?:~?/|[A-Za-z]:\\)[^\"']*[\"']"#, "[private path]"),
            (#"(?<![\p{L}\p{N}_./\\])(?:~?/|[A-Za-z]:\\)[^\r\n\"'<>]*"#, "[private path]"),
            (#"(?i)\#(credentialLabel)[\"']?\s*[:=]\s*[\"'][^\"']*[\"']"#, "[credential hidden]"),
            (#"(?i)\#(credentialLabel)[\"']?\s*[:=]\s*[\"']?[^\s,;]+"#, "[credential hidden]"),
            (#"(?i)\b(?:Bearer|Basic)\s+[^\s,;]+"#, "[credential hidden]"),
            (#"\b(?:sk-|ghp_|github_pat_|AIza|ya29\.|GOCSPX-|1//)[A-Za-z0-9_.-]+"#, "[credential hidden]"),
            (#"[A-Za-z0-9_+=.-]{32,}"#, "[token hidden]")
        ]
        for (pattern, replacement) in rules {
            text = text.replacingOccurrences(of: pattern, with: replacement, options: .regularExpression)
        }
        if indicatesFailure(input), !indicatesFailure(text) {
            text += " · Failure detail hidden for privacy; reveal log for details."
        }
        // Log-controlled newlines/control characters cannot imitate another receipt in the preview.
        return text.unicodeScalars.map { CharacterSet.controlCharacters.contains($0) ? " " : String($0) }.joined()
    }

    private static func indicatesFailure(_ text: String) -> Bool {
        text.range(of: #"(?i)\b(failed|failure|error|aborted|denied|timeout|timed\s+out|disk\s+full|not\s+verified)\b"#,
                   options: .regularExpression) != nil
    }
}

@MainActor
final class BrainBarBackupLogPreviewModel: ObservableObject {
    enum Kind: String, CaseIterable, Identifiable { case database = "Database", transcripts = "Transcripts"; var id: Self { self } }
    @Published var kind: Kind = .database
    @Published private(set) var preview: BrainBarBackupLogPreview = .unread
    @Published private(set) var refreshing = false
    private let paths: BrainBarBackupSources.Paths
    private let read: @Sendable (URL) -> BrainBarBackupLogPreview
    private let workspace: any BrainBarWorkspaceActing
    private var generation: UInt64 = 0

    init(paths: BrainBarBackupSources.Paths, workspace: any BrainBarWorkspaceActing = BrainBarWorkspace(),
         initialPreview: BrainBarBackupLogPreview = .unread,
         read: @escaping @Sendable (URL) -> BrainBarBackupLogPreview = { BrainBarBackupLogPreview.read($0) }) {
        self.paths = paths; self.workspace = workspace; self.read = read
        preview = initialPreview
    }

    private var url: URL { kind == .database ? paths.databaseLog : paths.archiveLog }

    func refresh() {
        generation &+= 1
        let ticket = generation, selected = kind, url = url, read = read
        refreshing = true
        preview = .unread
        Task {
            let result = await Task.detached { read(url) }.value
            guard ticket == generation, kind == selected else { return }
            preview = result
            refreshing = false
        }
    }

    func reveal() { workspace.reveal(url) }
}

struct BrainBarBackupLogPreviewView: View {
    @ObservedObject var model: BrainBarBackupLogPreviewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Recent backup log").font(.system(size: 14, weight: .semibold))
            Picker("Backup", selection: $model.kind) {
                ForEach(BrainBarBackupLogPreviewModel.Kind.allCases) { Text($0.rawValue).tag($0) }
            }
            .accessibilityIdentifier("brainbar.settings.backup-log.selector")
            ViewThatFits(in: .horizontal) {
                HStack { actions }.fixedSize()
                VStack(alignment: .leading, spacing: 6) { actions }.fixedSize()
            }
            Text("Latest 20 entries, up to 64 KB. Private paths and credentials are hidden; Finder opens the original log.")
                .font(.caption).foregroundStyle(Color.brainBarTextMuted)
                .fixedSize(horizontal: false, vertical: true)
            switch model.preview {
            case .unread:
                status(model.refreshing ? "Reading log…" : "Refresh to read the latest log.")
            case .empty: status("Log is empty. No backup entries recorded yet.")
            case let .unavailable(message): status(message)
            case let .entries(text, truncated):
                if truncated { status("Showing only the latest portion of the log.") }
                ScrollView {
                    Text(text).font(.system(size: 11, design: .monospaced))
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .accessibilityIdentifier("brainbar.settings.backup-log.content")
                }
                .frame(maxHeight: 240)
            }
        }
        .controlSize(.small)
        .task { model.refresh() }
        .onChange(of: model.kind) { _, _ in model.refresh() }
    }

    @ViewBuilder private var actions: some View {
        Button("Refresh") { model.refresh() }
            .disabled(model.refreshing)
            .accessibilityIdentifier("brainbar.settings.backup-log.refresh")
        Button("Reveal in Finder") { model.reveal() }
            .accessibilityIdentifier("brainbar.settings.backup-log.reveal")
    }

    private func status(_ text: String) -> some View {
        Text(text).font(.caption).fixedSize(horizontal: false, vertical: true)
            .accessibilityIdentifier("brainbar.settings.backup-log.status")
    }
}
