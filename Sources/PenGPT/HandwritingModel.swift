// Swift adaptation of the PenGPT handwriting sampling and layout code by Sam Greydanus and contributors.
// See NOTICE and LICENSE.

import Foundation

public struct InkPoint: Equatable, Sendable {
    public let x: Double
    public let y: Double

    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }
}

public struct InkStroke: Equatable, Sendable {
    /// Points in writing order, in model units with positive y pointing down.
    public let points: [InkPoint]

    public init(points: [InkPoint]) {
        self.points = points
    }
}

public struct InkDrawing: Equatable, Sendable {
    public let strokes: [InkStroke]

    public init(strokes: [InkStroke]) {
        self.strokes = strokes
    }
}

public struct HandwritingOptions: Sendable {
    /// Zero selects the most likely token. Positive values enable seeded sampling.
    public var temperature: Double
    public var topK: Int?
    public var seed: UInt32
    /// Maximum line width in model units. An individual word can extend past it.
    public var lineWidth: Double
    public var lineHeight: Double
    public var wordSpacing: Double
    /// Optional generation limit per decoder call, including any stop token.
    public var maxTokensPerChunk: Int?

    public init(
        temperature: Double = 0, topK: Int? = nil, seed: UInt32 = 0,
        lineWidth: Double = 8, lineHeight: Double = 0.55, wordSpacing: Double = 0.16,
        maxTokensPerChunk: Int? = nil
    ) {
        self.temperature = temperature
        self.topK = topK
        self.seed = seed
        self.lineWidth = lineWidth
        self.lineHeight = lineHeight
        self.wordSpacing = wordSpacing
        self.maxTokensPerChunk = maxTokensPerChunk
    }
}

public struct HandwritingModel: Sendable {
    public let writerIDs: [String]
    let decoder: Decoder

    /// Loads a local .pengpt file produced by pengpt-convert.
    public static func load(weightsAt url: URL) async throws -> Self {
        try Task.checkCancellation()
        return try Self(file: ModelFile.load(from: url))
    }

    package init(file: ModelFile) throws {
        let config = try file.configuration()
        guard config.format == 1, (config.vector ?? 0) == 0 else {
            throw PenGPTError.unsupportedModel
        }
        let stroke = try StrokeDecoder(grid: config.grid, merges: file.merges)
        decoder = try Decoder(file: file, config: config, stroke: stroke)
        writerIDs = file.writers
    }

    /// Generates laid-out ink. The first line's baseline is y = 0.
    public func write(_ text: String, writer: String = "", options: HandwritingOptions = .init()) async throws -> InkDrawing {
        try await collect(writeWords(text, writer: writer, options: options))
    }

    /// Generates laid-out ink using a learned writer embedding.
    public func write(_ text: String, style: HandwritingStyle, options: HandwritingOptions? = nil) async throws -> InkDrawing {
        try await collect(writeWords(text, style: style, options: options))
    }

    private func collect(_ words: HandwritingWords) async throws -> InkDrawing {
        var strokes: [InkStroke] = []
        for try await word in words {
            strokes.append(contentsOf: word.strokes)
        }
        return InkDrawing(strokes: strokes)
    }

    /// Creates a lazy sequence of laid-out ink, one element per whitespace-delimited word.
    ///
    /// Each element uses the same drawing coordinates as `write`. An element can be empty if
    /// generation fails to produce ink for its word. Each iterator starts a new generation.
    public func writeWords(_ text: String, writer: String = "", options: HandwritingOptions = .init()) throws -> HandwritingWords {
        guard let id = writerIDs.firstIndex(of: writer) else { throw PenGPTError.unknownWriter(writer) }
        return try words(text, style: decoder.writers.row(id), options: options)
    }

    /// Uses the fitted temperature and top-k 5 when sampling options are omitted, like the upstream photo command.
    public func writeWords(_ text: String, style: HandwritingStyle, options: HandwritingOptions? = nil) throws -> HandwritingWords {
        guard style.embedding.count == decoder.writers.columns, style.embedding.allSatisfy(\.isFinite),
              style.temperature.isFinite, style.temperature >= 0 else {
            throw PenGPTError.invalidOptions
        }
        return try words(text, style: style.embedding, options: options ?? .init(temperature: style.temperature, topK: 5))
    }

    private func words(_ text: String, style: [Float], options: HandwritingOptions) throws -> HandwritingWords {
        guard options.temperature.isFinite, options.temperature >= 0,
              (options.topK ?? 1) > 0, (options.maxTokensPerChunk ?? 1) > 0,
              options.lineWidth.isFinite, options.lineWidth > 0,
              options.lineHeight.isFinite, options.lineHeight > 0,
              options.wordSpacing.isFinite, options.wordSpacing >= 0 else {
            throw PenGPTError.invalidOptions
        }
        return HandwritingWords(
            decoder: decoder, words: text.split(whereSeparator: \.isWhitespace),
            style: style, options: options
        )
    }
}

/// Incremental handwriting. The decoder generates up to two words at a time and
/// advances only when the consumer requests more ink.
public struct HandwritingWords: AsyncSequence, Sendable {
    public typealias Element = InkDrawing

    fileprivate let decoder: Decoder
    fileprivate let words: [Substring]
    fileprivate let style: [Float]
    fileprivate let options: HandwritingOptions

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(self)
    }

    public struct AsyncIterator: AsyncIteratorProtocol, Sendable {
        private let decoder: Decoder
        private let style: [Float]
        private let options: HandwritingOptions
        private var remaining: ArraySlice<Substring>
        private var pending: ArraySlice<[Int]> = []
        private var rng: SamplingRandom
        private var previous: Substring?
        private var prefix: [Int] = []
        private var x = 0.0
        private var baseline = 0.0

        fileprivate init(_ sequence: HandwritingWords) {
            decoder = sequence.decoder
            style = sequence.style
            options = sequence.options
            remaining = sequence.words[...]
            rng = SamplingRandom(state: options.seed)
        }

        public mutating func next() async throws -> InkDrawing? {
            guard !pending.isEmpty || !remaining.isEmpty else { return nil }
            do {
                try Task.checkCancellation()
                if pending.isEmpty { try generateChunk() }
                return InkDrawing(strokes: place(decoder.stroke.decode(pending.removeFirst())))
            } catch {
                remaining = []
                pending = []
                throw error
            }
        }

        private mutating func generateChunk() throws {
            let chunk = remaining.prefix(2)
            let input = chunk.joined(separator: " ")
            let continued = previous.map { $0 + " " + input } ?? input
            let reusePrefix = continued.utf8.count <= decoder.maxTextLength
                && prefix.count + 3 < decoder.maxSequence
            var generated = try tokenWords(reusePrefix ? continued : input, prefix: reusePrefix ? prefix : [])
            for word in chunk.dropFirst(generated.count) {
                let fallback = try tokenWords(String(word), prefix: [])
                generated.append(fallback.first ?? [])
            }
            pending = generated.prefix(chunk.count)
            remaining = remaining.dropFirst(chunk.count)
            if pending.allSatisfy({ !$0.isEmpty }) {
                previous = chunk.last
                prefix = pending.last ?? []
            } else {
                previous = nil
                prefix = []
            }
        }

        private mutating func tokenWords(_ text: String, prefix: [Int]) throws -> [[Int]] {
            let raw = try decoder.generate(text, writer: 0, prefix: prefix, options: options, rng: &rng,
                                           guidance: decoder.labelDrop ? 2 : 1, style: style)
            let complete = raw.last == decoder.stroke.end
            let body = complete ? raw.dropLast() : raw[...]
            let words = body.split(separator: decoder.stroke.word).map(Array.init)
            return complete ? words : Array(words.dropLast())
        }

        private mutating func place(_ paths: [[Coordinate]]) -> [InkStroke] {
            guard let first = paths.first?.first else { return [] }
            var minX = first.x
            var maxX = first.x
            for path in paths {
                for point in path {
                    minX = Swift.min(minX, point.x)
                    maxX = Swift.max(maxX, point.x)
                }
            }
            let width = maxX - minX
            if x > 0 && x + width > options.lineWidth {
                x = 0
                baseline += options.lineHeight
            }
            let strokes = paths.map { path in
                InkStroke(points: path.map { .init(x: $0.x - minX + x, y: $0.y + baseline) })
            }
            x += width + options.wordSpacing
            return strokes
        }
    }
}
