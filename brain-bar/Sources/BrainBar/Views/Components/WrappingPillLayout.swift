import SwiftUI

struct WrappingPillLayout: Layout {
    let spacing: CGFloat
    let lineSpacing: CGFloat

    init(spacing: CGFloat = 8, lineSpacing: CGFloat = 8) {
        self.spacing = spacing
        self.lineSpacing = lineSpacing
    }

    func sizeThatFits(
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout Void
    ) -> CGSize {
        let maxWidth = proposal.width ?? subviews.enumerated().reduce(CGFloat.zero) { width, pair in
            width + pair.element.sizeThatFits(.unspecified).width + (pair.offset > 0 ? spacing : 0)
        }
        let rows = makeRows(subviews: subviews, maxWidth: max(maxWidth, 1))
        let height = rows.reduce(CGFloat.zero) { total, row in
            total + row.height
        } + CGFloat(max(rows.count - 1, 0)) * lineSpacing
        let width = min(maxWidth, rows.map(\.width).max() ?? 0)
        return CGSize(width: width, height: height)
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout Void
    ) {
        let rows = makeRows(subviews: subviews, maxWidth: max(bounds.width, 1))
        var y = bounds.minY

        for row in rows {
            var x = bounds.minX
            for item in row.items {
                subviews[item.index].place(
                    at: CGPoint(x: x, y: y),
                    anchor: .topLeading,
                    proposal: ProposedViewSize(width: item.size.width, height: item.size.height)
                )
                x += item.size.width + spacing
            }
            y += row.height + lineSpacing
        }
    }

    private func makeRows(subviews: Subviews, maxWidth: CGFloat) -> [Row] {
        var rows: [Row] = []
        var currentItems: [Item] = []
        var currentWidth: CGFloat = 0
        var currentHeight: CGFloat = 0

        for index in subviews.indices {
            let proposedSize = subviews[index].sizeThatFits(
                ProposedViewSize(width: maxWidth, height: nil)
            )
            let itemWidth = min(proposedSize.width, maxWidth)
            let item = Item(
                index: index,
                size: CGSize(width: itemWidth, height: proposedSize.height)
            )
            let additionalWidth = currentItems.isEmpty ? itemWidth : spacing + itemWidth

            if !currentItems.isEmpty, currentWidth + additionalWidth > maxWidth {
                rows.append(Row(items: currentItems, width: currentWidth, height: currentHeight))
                currentItems = [item]
                currentWidth = itemWidth
                currentHeight = item.size.height
            } else {
                currentItems.append(item)
                currentWidth += additionalWidth
                currentHeight = max(currentHeight, item.size.height)
            }
        }

        if !currentItems.isEmpty {
            rows.append(Row(items: currentItems, width: currentWidth, height: currentHeight))
        }
        return rows
    }

    private struct Item {
        let index: Int
        let size: CGSize
    }

    private struct Row {
        let items: [Item]
        let width: CGFloat
        let height: CGFloat
    }
}
