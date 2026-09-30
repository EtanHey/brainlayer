import AppKit
import SwiftUI

// Reconnect Google Drive: the Backups page card and the Dashboard banner over BrainBarDriveAuthModel.

private struct BrainBarDriveAuthKey: EnvironmentKey {
    static let defaultValue: BrainBarDriveAuthModel? = nil
}

extension EnvironmentValues {
    /// The shared Drive-access model. Nil outside the BrainBar window, so a view hosted anywhere
    /// else shows no Drive block and never runs the CLI.
    var brainBarDriveAuth: BrainBarDriveAuthModel? {
        get { self[BrainBarDriveAuthKey.self] }
        set { self[BrainBarDriveAuthKey.self] = newValue }
    }
}

extension DriveAuthPresentation.Tone {
    var color: Color {
        let color: NSColor = switch self {
        case .connected: BrainBarDesignTokens.Colors.statusOK
        case .expiring: BrainBarDesignTokens.Colors.statusAttention
        case .attention: BrainBarDesignTokens.Colors.statusError
        case .unknown: BrainBarDesignTokens.Colors.statusUnknown
        }
        return Color(nsColor: color)
    }
}

enum BrainBarDriveAuthFormat {
    static func date(_ date: Date) -> String {
        date.formatted(.dateTime.weekday(.abbreviated).month(.abbreviated).day().hour().minute())
    }
}

/// The Backups page's Google Drive block: state, what to do, and the one Reconnect button.
struct BrainBarDriveAuthCard: View {
    @ObservedObject var model: BrainBarDriveAuthModel

    var body: some View {
        let presentation = model.presentation(formatDate: BrainBarDriveAuthFormat.date)
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Circle().fill(presentation.tone.color).frame(width: 8, height: 8)
                Text("Google Drive").font(.system(size: 14, weight: .semibold))
                Spacer(minLength: 12)
                BrainBarDriveReconnectButton(model: model, presentation: presentation)
            }
            Text(presentation.line)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(presentation.tone == .connected ? Color.brainBarTextSecondary : presentation.tone.color)
                .fixedSize(horizontal: false, vertical: true)
            if let detail = presentation.detail {
                Text(detail)
                    .font(.system(size: 11))
                    .foregroundStyle(Color.brainBarTextMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("brainbar.backups.google-drive")
        .task { await model.refreshIfStale() }
    }
}

/// The Dashboard banner, shown only when Drive needs a click (missing, invalid, or expiring).
struct BrainBarDriveAuthBanner: View {
    @ObservedObject var model: BrainBarDriveAuthModel

    var body: some View {
        let presentation = model.presentation(formatDate: BrainBarDriveAuthFormat.date)
        Group {
            if presentation.needsAttention {
                HStack(alignment: .center, spacing: 10) {
                    Image(systemName: "externaldrive.badge.exclamationmark")
                        .foregroundStyle(presentation.tone.color)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(presentation.line)
                            .font(.system(size: 12, weight: .semibold))
                        if let detail = presentation.detail {
                            Text(detail)
                                .font(.system(size: 11))
                                .foregroundStyle(Color.brainBarTextSecondary)
                                .lineLimit(2)
                        }
                    }
                    Spacer(minLength: 8)
                    BrainBarDriveReconnectButton(model: model, presentation: presentation)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(presentation.tone.color.opacity(0.10))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(presentation.tone.color.opacity(0.35), lineWidth: 1)
                )
                .accessibilityElement(children: .contain)
                .accessibilityIdentifier("brainbar.dashboard.google-drive-banner")
            }
        }
        .task { await model.refreshIfStale() }
    }
}

private struct BrainBarDriveReconnectButton: View {
    @ObservedObject var model: BrainBarDriveAuthModel
    let presentation: DriveAuthPresentation

    var body: some View {
        if presentation.showsReconnect {
            HStack(spacing: 6) {
                if model.isReconnecting {
                    ProgressView().controlSize(.small)
                }
                Button(DriveAuthPresentation.buttonTitle) {
                    Task { await model.reconnect() }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(!presentation.reconnectEnabled)
                .help("Opens Google consent in your browser. BrainBar never sees or shows the token.")
            }
        }
    }
}
