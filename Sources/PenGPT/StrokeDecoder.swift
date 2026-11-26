// Swift adaptation of the PenGPT stroke decoder and curve code by Sam Greydanus and contributors.
// See NOTICE and LICENSE.

import Foundation

typealias Coordinate = SIMD2<Double>

struct StrokeDecoder: Sendable {
    let grid: Double
    let paths: [[SIMD2<Int>]]
    let deltas: [SIMD2<Int>]

    var pad: Int { paths.count - 4 }
    var end: Int { pad + 1 }
    var word: Int { pad + 2 }
    var bos: Int { pad + 3 }

    init(grid: Double, merges: [Int]) throws {
        guard merges.count.isMultiple(of: 3), merges.count <= 12_288 else {
            throw PenGPTError.invalidModel("stroke merges")
        }
        let directions: [SIMD2<Int>] = [
            .init(1, 0), .init(1, 1), .init(0, 1), .init(-1, 1),
            .init(-1, 0), .init(-1, -1), .init(0, -1), .init(1, -1),
        ]
        var paths = directions.map { [$0] } + [[], []]
        var deltas = directions + [.zero, .zero]
        for offset in stride(from: 0, to: merges.count, by: 3) {
            let a = merges[offset]
            let b = merges[offset + 1]
            guard paths.indices.contains(a), paths.indices.contains(b), merges[offset + 2] == paths.count,
                  paths[a].count + paths[b].count <= 4096 else {
                throw PenGPTError.invalidModel("stroke merge at \(offset / 3)")
            }
            let shift = deltas[a]
            paths.append(paths[a] + paths[b].map { $0 &+ shift })
            deltas.append(shift &+ deltas[b])
        }
        self.grid = grid
        self.paths = paths + [[], [], [], []]
        self.deltas = deltas + [.zero, .zero, .zero, .zero]
    }

    func decode(_ tokens: [Int]) -> [[Coordinate]] {
        var position = SIMD2<Int>.zero
        var polylines: [[Coordinate]] = []
        var current: [Coordinate] = []
        func point(_ xy: SIMD2<Int>) -> Coordinate {
            Coordinate(Double(xy.x) * grid, Double(xy.y) * grid)
        }

        for token in tokens {
            switch token {
            case 8:
                current.append(point(position))
            case 9:
                if !current.isEmpty {
                    polylines.append(smooth(current, grid: grid))
                    current.removeAll(keepingCapacity: true)
                }
            default:
                if !current.isEmpty {
                    current.append(contentsOf: paths[token].map { point(position &+ $0) })
                }
                position &+= deltas[token]
            }
        }
        if !current.isEmpty { polylines.append(smooth(current, grid: grid)) }
        return polylines
    }
}

private func smooth(_ original: [Coordinate], grid: Double) -> [Coordinate] {
    guard original.count >= 3 else { return original }
    var points = original
    for i in 1 ..< points.count - 1 {
        points[i] = (original[i - 1] + original[i] + original[i + 1]) / 3
    }
    let unique = points.indices.compactMap { i in
        i == 0 || points[i] != points[i - 1] ? points[i] : nil
    }
    guard unique.count >= 3 else { return points }

    let corners = reduceCorners(unique, tolerance: grid / 2)
    var cuts = [0]
    for i in 1 ..< corners.count - 1 {
        if isSharp(corners[i] - corners[i - 1], corners[i + 1] - corners[i]) { cuts.append(i) }
    }
    let last = corners.count - 1
    cuts.append(last)
    let closed = corners.count > 4 && cuts.count == 2 && length(corners[last] - corners[0]) < grid
        && !isSharp(corners[0] - corners[last - 1], corners[1] - corners[0])
    let pieces: [[Coordinate]]
    if closed {
        pieces = [chaikin(Array(corners.dropLast()), closed: true)]
    } else {
        pieces = zip(cuts, cuts.dropFirst()).map { chaikin(Array(corners[$0 ... $1]), closed: false) }
    }
    let distances = pieces.map(cumulativeLengths)
    let total = distances.reduce(0) { $0 + $1.last! }
    guard total > 0 else { return points }

    var end = 0.0
    var previous = 0
    var counts = distances.map { distance in
        end += distance.last!
        let rounded = Int((end / total * Double(points.count - 1)).rounded())
        defer { previous = rounded }
        return max(rounded - previous, 1)
    }
    var excess = counts.reduce(0, +) - (points.count - 1)
    while excess > 0 {
        let most = counts.indices.max { counts[$0] < counts[$1] }!
        counts[most] -= 1
        excess -= 1
    }

    return pieces.indices.flatMap { i in
        resample(pieces[i], at: distances[i], segments: counts[i]).dropFirst(i == 0 ? 0 : 1)
    }
}

private func reduceCorners(_ points: [Coordinate], tolerance: Double) -> [Coordinate] {
    var keep = Set([0, points.count - 1])
    var pending = [(0, points.count - 1)]
    while let (start, end) = pending.popLast() {
        guard end > start + 1 else { continue }
        let axis = points[end] - points[start]
        let squared = dot(axis, axis)
        var furthest: Int?
        var distance = tolerance
        for i in start + 1 ..< end {
            let offset = points[i] - points[start]
            let projection = squared > 0 ? min(1, max(0, dot(offset, axis) / squared)) : 0
            let error = length(offset - axis * projection)
            if error > distance {
                furthest = i
                distance = error
            }
        }
        if let furthest {
            keep.insert(furthest)
            pending.append((start, furthest))
            pending.append((furthest, end))
        }
    }
    return keep.sorted().map { points[$0] }
}

private func chaikin(_ points: [Coordinate], closed: Bool) -> [Coordinate] {
    var curve = points
    for _ in 0 ..< 4 where curve.count > 2 {
        var cut: [Coordinate] = []
        let edges = curve.count - (closed ? 0 : 1)
        for i in 0 ..< edges {
            let a = curve[i]
            let b = curve[(i + 1) % curve.count]
            cut.append(0.75 * a + 0.25 * b)
            cut.append(0.25 * a + 0.75 * b)
        }
        if !closed {
            cut[0] = curve[0]
            cut[cut.count - 1] = curve.last!
        }
        curve = cut
    }
    if closed { curve.append(curve[0]) }
    return curve
}

private func cumulativeLengths(_ points: [Coordinate]) -> [Double] {
    var distances = [0.0]
    for i in 1 ..< points.count {
        distances.append(distances[i - 1] + length(points[i] - points[i - 1]))
    }
    return distances
}

private func resample(_ points: [Coordinate], at distances: [Double], segments: Int) -> [Coordinate] {
    var i = 0
    return (0 ... segments).map { sample in
        let target = distances.last! * Double(sample) / Double(segments)
        while i < points.count - 2 && distances[i + 1] < target { i += 1 }
        let span = distances[i + 1] - distances[i]
        let fraction = span > 0 ? min(1, (target - distances[i]) / span) : 0
        return points[i] + fraction * (points[i + 1] - points[i])
    }
}

private func isSharp(_ a: Coordinate, _ b: Coordinate) -> Bool {
    dot(a, b) <= -0.5 * sqrt(dot(a, a) * dot(b, b))
}

private func length(_ x: Coordinate) -> Double { sqrt(dot(x, x)) }

private func dot(_ a: Coordinate, _ b: Coordinate) -> Double { a.x * b.x + a.y * b.y }
