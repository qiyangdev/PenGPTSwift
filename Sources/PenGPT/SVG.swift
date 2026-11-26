import Foundation

extension InkDrawing {
    /// Encodes ordered pen strokes as SVG paths. Scale and padding are expressed in SVG units
    /// per model unit and in model units respectively. Colors use six-digit hexadecimal RGB.
    public func svg(
        scale: Double = 160, padding: Double = 0.3, strokeWidth: Double = 2.2,
        ink: String = "#1a237e", background: String? = "#ffffff"
    ) throws -> String {
        guard scale.isFinite, scale > 0, padding.isFinite, padding >= 0,
            strokeWidth.isFinite, strokeWidth > 0, isRGB(ink),
            background.map(isRGB) ?? true
        else {
            throw PenGPTError.invalidOptions
        }
        var minX = Double.infinity
        var minY = Double.infinity
        var maxX = -Double.infinity
        var maxY = -Double.infinity
        for point in strokes.lazy.flatMap(\.points) {
            guard point.x.isFinite, point.y.isFinite else { throw PenGPTError.invalidOptions }
            minX = min(minX, point.x)
            minY = min(minY, point.y)
            maxX = max(maxX, point.x)
            maxY = max(maxY, point.y)
        }
        if strokes.allSatisfy({ $0.points.isEmpty }) {
            minX = 0
            minY = 0
            maxX = 0
            maxY = 0
        }
        let originX = minX - padding
        let originY = minY - padding
        let width = max(strokeWidth, (maxX - minX + 2 * padding) * scale)
        let height = max(strokeWidth, (maxY - minY + 2 * padding) * scale)
        guard width.isFinite, height.isFinite, originX.isFinite, originY.isFinite else {
            throw PenGPTError.invalidOptions
        }
        let dimensions = "\(width) \(height)"
        var svg =
            #"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 \#(dimensions)" width="\#(width)" height="\#(height)">"#
        if let background {
            svg += #"<rect width="100%" height="100%" fill="\#(background)"/>"#
        }
        svg +=
            #"<g fill="none" stroke="\#(ink)" stroke-width="\#(strokeWidth)" stroke-linecap="round" stroke-linejoin="round">"#
        for stroke in strokes {
            guard let first = stroke.points.first else { continue }
            func coordinate(_ point: InkPoint) -> String {
                "\((point.x - originX) * scale) \((point.y - originY) * scale)"
            }
            let start = coordinate(first)
            let rest = stroke.points.dropFirst().map(coordinate)
            svg += #"<path d="M\#(start) L\#(rest.isEmpty ? start : rest.joined(separator: " L"))"/>"#
        }
        return svg + "</g></svg>"
    }
}

private func isRGB(_ value: String) -> Bool {
    let bytes = value.utf8
    return bytes.count == 7 && bytes.first == 35
        && bytes.dropFirst().allSatisfy {
            (48 ... 57).contains($0) || (65 ... 70).contains($0) || (97 ... 102).contains($0)
        }
}
