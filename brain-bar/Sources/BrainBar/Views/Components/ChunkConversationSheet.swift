import SwiftUI

struct InjectionThreadRecipient: Equatable, Sendable {
    let sessionName: String
    let agentName: String
    let projectName: String
    let sessionID: String

    init(sessionName: String, agentName: String, projectName: String, sessionID: String) {
        self.sessionName = sessionName.trimmingCharacters(in: .whitespacesAndNewlines)
        self.agentName = agentName.trimmingCharacters(in: .whitespacesAndNewlines)
        self.projectName = projectName.trimmingCharacters(in: .whitespacesAndNewlines)
        self.sessionID = sessionID.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    init(event: InjectionEvent) {
        self.init(sessionName: event.sessionName, agentName: event.agentName, projectName: event.projectName, sessionID: event.sessionID)
    }

    var displayText: String {
        var parts: [String] = []
        if !sessionName.isEmpty { parts.append("Session \(sessionName)") }
        if !agentName.isEmpty { parts.append("Agent \(agentName)") }
        if !projectName.isEmpty { parts.append("Project \(projectName)") }
        if !parts.isEmpty { return parts.joined(separator: " · ") }
        if sessionID.isEmpty { return "Session unavailable" }
        return sessionID.count > 12 ? "Session …\(sessionID.suffix(6))" : "Session \(sessionID)"
    }
}

struct ChunkConversationSheet: View {
    let conversation: BrainDatabase.ExpandedConversation
    let title: String
    let recipient: InjectionThreadRecipient?
    let onClose: (() -> Void)?
    @Environment(\.dismiss) private var dismiss
    @State private var showsRawText = false

    init(
        conversation: BrainDatabase.ExpandedConversation,
        title: String = "Conversation",
        recipient: InjectionThreadRecipient? = nil,
        onClose: (() -> Void)? = nil
    ) {
        self.conversation = conversation
        self.title = title
        self.recipient = recipient
        self.onClose = onClose
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            identityStrip
            Picker("Thread text", selection: $showsRawText) {
                Text("Rendered").tag(false)
                Text("Raw text").tag(true)
            }
            .pickerStyle(.segmented)
            .frame(width: 210)
            .accessibilityLabel("Conversation text presentation")

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(conversation.entries) { entry in
                        conversationTurn(entry)
                    }
                }
            }
            technicalDetails
        }
        .padding(18)
        .frame(minWidth: 580, minHeight: 460)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Full conversation")
                    .font(.title2.bold())
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                Text(originText)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(conversation.origin == .sourceJSONL ? .green : .orange)
            }
            Spacer()
            Button("Close", action: close)
        }
    }

    private var identityStrip: some View {
        VStack(alignment: .leading, spacing: 7) {
            if let recipient {
                Text("Received by")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                Text(recipient.displayText)
                    .font(.system(size: 12, weight: .semibold))
            }
            Text("Stored by \(storingAgentText)")
                .font(.system(size: 12, weight: .semibold))
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Color.brainBarTextPrimary.opacity(0.055)))
    }

    private func conversationTurn(_ entry: BrainDatabase.ConversationChunk) -> some View {
        let role = rolePresentation(for: entry)
        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text(role.label)
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(role.tint)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 4)
                    .background(Capsule().fill(role.tint.opacity(0.16)))
                if entry.isTarget {
                    Text("Injected memory")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.blue)
                }
                Spacer()
                Text(String(entry.createdAt.prefix(19)))
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            if !entry.summary.isEmpty {
                Text(entry.summary)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
            }
            if showsRawText {
                Text(entry.content)
                    .font(.system(size: 13, design: .monospaced))
                    .textSelection(.enabled)
            } else {
                ConversationMarkdownView(markdown: entry.content)
                    .textSelection(.enabled)
            }
        }
        .padding(13)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(role.tint.opacity(entry.isTarget ? 0.13 : 0.075)))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(role.tint.opacity(entry.isTarget ? 0.48 : 0.24), lineWidth: entry.isTarget ? 1.5 : 1)
        )
    }

    private var technicalDetails: some View {
        DisclosureGroup("Technical details") {
            VStack(alignment: .leading, spacing: 5) {
                Text("Target ID \(conversation.target.chunkID)")
                if !conversation.sourceFile.isEmpty { Text("JSONL \(conversation.sourceFile)") }
                if !conversation.target.tags.isEmpty { Text("Tags \(conversation.target.tags.joined(separator: ", "))") }
            }
            .font(.system(size: 10, weight: .medium, design: .monospaced))
            .foregroundStyle(.secondary)
            .textSelection(.enabled)
            .padding(.top, 6)
        }
        .font(.system(size: 11, weight: .semibold))
    }

    private var originText: String {
        switch conversation.origin {
        case .sourceJSONL: return "Source JSONL · \(conversation.entries.count) turns"
        case .indexedExcerpt: return "Indexed excerpt · source JSONL unavailable"
        }
    }

    private var storingAgentText: String {
        let agent = conversation.storingAgent.trimmingCharacters(in: .whitespacesAndNewlines)
        return agent.isEmpty ? "Agent unavailable" : "Agent \(agent)"
    }

    private func rolePresentation(for entry: BrainDatabase.ConversationChunk) -> (label: String, tint: Color) {
        switch entry.sender.lowercased() {
        case "user": return ("You", .blue)
        case "assistant": return (storingAgentText, .brainBarAccentViolet)
        default: break
        }
        switch entry.contentType {
        case "user_message": return ("You", .blue)
        case "assistant_text": return (storingAgentText, .brainBarAccentViolet)
        default: return (entry.contentType.replacingOccurrences(of: "_", with: " ").capitalized, .secondary)
        }
    }

    private func close() {
        if let onClose { onClose() } else { dismiss() }
    }
}

private struct ConversationMarkdownView: View {
    let markdown: String

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            ForEach(Array(MarkdownBlock.parse(markdown).enumerated()), id: \.offset) { _, block in
                blockView(block)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func blockView(_ block: MarkdownBlock) -> some View {
        switch block {
        case .heading(let level, let text):
            inlineText(text).font(.system(size: level == 1 ? 20 : level == 2 ? 17 : 15, weight: .bold, design: .rounded))
        case .paragraph(let text):
            inlineText(text).font(.system(size: 13))
        case .bullet(let text, let checked):
            HStack(alignment: .firstTextBaseline, spacing: 7) {
                if let checked {
                    Image(systemName: checked ? "checkmark.square.fill" : "square")
                        .foregroundStyle(checked ? Color.green : Color.secondary)
                } else {
                    Text("•")
                }
                inlineText(text).font(.system(size: 13))
            }
        case .numbered(let number, let text):
            HStack(alignment: .firstTextBaseline, spacing: 7) {
                Text("\(number).").font(.system(size: 13, weight: .semibold, design: .monospaced))
                inlineText(text).font(.system(size: 13))
            }
        case .code(let code):
            ScrollView(.horizontal) {
                Text(code).font(.system(size: 12, design: .monospaced)).padding(10).textSelection(.enabled)
            }
            .background(RoundedRectangle(cornerRadius: 8, style: .continuous).fill(Color.brainBarBlack.opacity(0.28)))
        }
    }

    private func inlineText(_ markdown: String) -> Text {
        let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        if let attributed = try? AttributedString(markdown: markdown, options: options) {
            return Text(attributed)
        }
        return Text(markdown)
    }
}

private enum MarkdownBlock {
    case heading(level: Int, text: String)
    case paragraph(String)
    case bullet(text: String, checked: Bool?)
    case numbered(number: Int, text: String)
    case code(String)

    static func parse(_ markdown: String) -> [MarkdownBlock] {
        let lines = markdown.components(separatedBy: .newlines)
        var blocks: [MarkdownBlock] = []
        var paragraph: [String] = []
        var code: [String] = []
        var insideCode = false

        func flushParagraph() {
            guard !paragraph.isEmpty else { return }
            blocks.append(.paragraph(paragraph.joined(separator: "\n")))
            paragraph.removeAll()
        }
        func flushCode() {
            blocks.append(.code(code.joined(separator: "\n")))
            code.removeAll()
        }

        for line in lines {
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                if insideCode { flushCode() } else { flushParagraph() }
                insideCode.toggle()
                continue
            }
            if insideCode { code.append(line); continue }
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { flushParagraph(); continue }
            if let heading = heading(from: trimmed) {
                flushParagraph(); blocks.append(.heading(level: heading.level, text: heading.text)); continue
            }
            if let checklist = checklist(from: trimmed) {
                flushParagraph(); blocks.append(.bullet(text: checklist.text, checked: checklist.checked)); continue
            }
            if let bullet = bullet(from: trimmed) {
                flushParagraph(); blocks.append(.bullet(text: bullet, checked: nil)); continue
            }
            if let numbered = numbered(from: trimmed) {
                flushParagraph(); blocks.append(.numbered(number: numbered.number, text: numbered.text)); continue
            }
            paragraph.append(line)
        }
        if insideCode { flushCode() }
        flushParagraph()
        return blocks.isEmpty ? [.paragraph(markdown)] : blocks
    }

    private static func heading(from line: String) -> (level: Int, text: String)? {
        let count = line.prefix(while: { $0 == "#" }).count
        guard (1...6).contains(count), line.dropFirst(count).first == " " else { return nil }
        return (count, String(line.dropFirst(count + 1)))
    }

    private static func checklist(from line: String) -> (text: String, checked: Bool)? {
        for (prefix, checked) in [("- [x] ", true), ("- [X] ", true), ("- [ ] ", false)] where line.hasPrefix(prefix) {
            return (String(line.dropFirst(prefix.count)), checked)
        }
        return nil
    }

    private static func bullet(from line: String) -> String? {
        for prefix in ["- ", "* ", "+ "] where line.hasPrefix(prefix) {
            return String(line.dropFirst(prefix.count))
        }
        return nil
    }

    private static func numbered(from line: String) -> (number: Int, text: String)? {
        guard let marker = line.firstIndex(where: { $0 == "." || $0 == ")" }),
              line.index(after: marker) < line.endIndex,
              line[line.index(after: marker)] == " ",
              let number = Int(line[..<marker]) else { return nil }
        return (number, String(line[line.index(marker, offsetBy: 2)...]))
    }
}

struct ChunkConversationOverlay: View {
    let conversation: BrainDatabase.ExpandedConversation
    let title: String
    let recipient: InjectionThreadRecipient?
    let onClose: () -> Void

    init(
        conversation: BrainDatabase.ExpandedConversation,
        title: String = "Conversation",
        recipient: InjectionThreadRecipient? = nil,
        onClose: @escaping () -> Void
    ) {
        self.conversation = conversation
        self.title = title
        self.recipient = recipient
        self.onClose = onClose
    }

    var body: some View {
        ZStack {
            Rectangle().fill(Color.brainBarBlack.opacity(0.5)).contentShape(Rectangle()).onTapGesture(perform: onClose)
            ChunkConversationSheet(conversation: conversation, title: title, recipient: recipient, onClose: onClose)
                .background(
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .fill(.regularMaterial)
                        .overlay(RoundedRectangle(cornerRadius: 24, style: .continuous).fill(Color.brainBarGlassPrimary))
                )
                .overlay(RoundedRectangle(cornerRadius: 24, style: .continuous).stroke(Color.brainBarTextPrimary.opacity(0.08), lineWidth: 1))
                .shadow(color: Color.brainBarBlack.opacity(0.18), radius: 24, y: 10)
                .padding(28)
                .frame(maxWidth: 940, maxHeight: .infinity)
                .onTapGesture {}
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .transition(.opacity)
        .zIndex(30)
    }
}
