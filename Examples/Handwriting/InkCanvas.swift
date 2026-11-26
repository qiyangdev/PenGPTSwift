import PenGPT
import SwiftUI

struct InkCanvas: View {
    let strokes: [[InkPoint]]
    let extent: CGRect
    let lineWidth: Double
    let isSketch: Bool

    @Environment(\.colorScheme) private var colorScheme

    private static let inset: CGFloat = 20
    private static let maxScale: CGFloat = 130
    // Reserve the upstream handwriting viewport's headroom above the first baseline.
    private static let handwritingBaseline: CGFloat = inset + 0.42 * maxScale
    // The upstream sketch preview uses a fixed viewBox from -0.62 to 0.62 on each axis.
    private static let sketchSpan: CGFloat = 1.24

    static func handwritingLineWidth(for width: CGFloat) -> Double {
        max(1, (width - 2 * inset) / maxScale)
    }

    var body: some View {
        Canvas { context, size in
            let inset = Self.inset
            let usableWidth = max(1, size.width - 2 * inset)
            let scale: CGFloat
            let origin: CGPoint
            if isSketch {
                let usableHeight = max(1, size.height - 2 * inset)
                scale = min(usableWidth, usableHeight) / Self.sketchSpan
                origin = CGPoint(x: size.width / 2, y: size.height / 2)
            } else {
                scale = min(Self.maxScale, usableWidth / max(lineWidth, extent.width))
                origin = CGPoint(x: inset - extent.minX * scale, y: Self.handwritingBaseline)
            }
            let ink =
                colorScheme == .light
                ? Color(red: 0.12, green: 0.20, blue: 0.38)
                : Color(red: 0.65, green: 0.77, blue: 1)
            func position(_ point: InkPoint) -> CGPoint {
                CGPoint(x: origin.x + point.x * scale, y: origin.y + point.y * scale)
            }

            for points in strokes {
                guard let first = points.first else { continue }
                let origin = position(first)
                if points.count == 1 {
                    let dot = CGRect(x: origin.x - 1, y: origin.y - 1, width: 2, height: 2)
                    context.fill(Path(ellipseIn: dot), with: .color(ink))
                } else {
                    var path = Path()
                    path.move(to: origin)
                    path.addLines(points.dropFirst().map(position))
                    context.stroke(
                        path, with: .color(ink),
                        style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
                }
            }
        }
        .frame(maxWidth: .infinity)
        .frame(
            height: isSketch
                ? 240
                : max(
                    240, Self.handwritingBaseline + max(0, extent.maxY) * Self.maxScale + Self.inset
                )
        )
        .accessibilityLabel("Handwriting output")
    }
}
