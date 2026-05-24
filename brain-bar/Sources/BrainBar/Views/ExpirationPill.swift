import SwiftUI

struct ExpirationPill: View {
    let date: Date
    let label: String

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "clock.arrow.circlepath")
                .font(.system(size: 9, weight: .medium))
            Text(Self.displayText(date: date, label: label))
                .font(.system(size: 11, weight: .medium))
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(Capsule().fill(Color.secondary.opacity(0.12)))
    }

    nonisolated static func displayText(date: Date, label: String) -> String {
        "\(label) \(formattedDate(date))"
    }

    nonisolated static func formattedDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "MMM d, yyyy"
        return formatter.string(from: date)
    }
}
