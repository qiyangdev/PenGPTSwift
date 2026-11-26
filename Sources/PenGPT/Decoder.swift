// Swift adaptation of the PenGPT NumPy and browser decoders by Sam Greydanus and contributors.
// See NOTICE and LICENSE.

import Accelerate
import Foundation

struct Matrix: Sendable {
    let rows: Int
    let columns: Int
    let values: [Float]

    func row(_ index: Int) -> [Float] {
        let offset = index * columns
        return Array(values[offset ..< offset + columns])
    }

    func multiply(_ input: [Float]) -> [Float] {
        var output = [Float](repeating: 0, count: rows)
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(rows), Int32(columns), 1,
                    values, Int32(columns), input, 1, 0, &output, 1)
        return output
    }
}

struct Attention: Sendable {
    let q: Matrix
    let kv: Matrix
    let projection: Matrix
    let qNorm: [Float]
    let kNorm: [Float]
}

struct DecoderLayer: Sendable {
    let selfAttention: Attention
    let crossAttention: Attention
    let feedForward: Matrix
    let feedForwardOutput: Matrix
    let residualScale: Float
    let embeddingScale: Float
}

struct Decoder: Sendable {
    let numHeads: Int
    let maxSequence: Int
    let maxTextLength: Int
    let textEnd: Bool
    let labelDrop: Bool
    let hasHints: Bool
    let penRotary: Bool
    let stroke: StrokeDecoder
    let tokens: Matrix
    let writers: Matrix
    let contextTokens: Matrix
    let contextPositions: Matrix
    let penProjection: Matrix?
    let vectorSize: Int
    let vectorProjection: Matrix?
    private let vectorSlot: [Float]
    private let hintProjection: Matrix?
    let layers: [DecoderLayer]
    let head: Matrix

    private let positionFrequencies: [Float]
    private let rotaryFrequencies: [Float]

    init(file: ModelFile, config: ModelConfiguration, stroke: StrokeDecoder) throws {
        let embedding = try file.matrix("tok_emb.weight", rows: stroke.paths.count)
        let width = embedding.columns
        guard width.isMultiple(of: 2 * config.nHead) else {
            throw PenGPTError.invalidModel("attention dimensions")
        }
        let headWidth = width / config.nHead
        let usesPenRotary = config.penRotary ?? false
        guard !usesPenRotary || headWidth.isMultiple(of: 8) else { throw PenGPTError.unsupportedModel }

        func attention(_ prefix: String) throws -> Attention {
            try Attention(
                q: file.matrix(prefix + ".q.weight", rows: width, columns: width),
                kv: file.matrix(prefix + ".kv.weight", rows: 2 * width, columns: width),
                projection: file.matrix(prefix + ".proj.weight", rows: width, columns: width),
                qNorm: file.vector(prefix + ".q_norm.weight", headWidth),
                kNorm: file.vector(prefix + ".k_norm.weight", headWidth)
            )
        }

        numHeads = config.nHead
        maxSequence = min(config.maxSeqLength, config.blockSize)
        maxTextLength = config.maxTextLength
        textEnd = config.textEnd
        labelDrop = (config.labelDrop ?? 0) > 0
        hasHints = (config.hint ?? 0) > 0
        penRotary = usesPenRotary
        self.stroke = stroke
        tokens = embedding
        writers = try file.matrix("writer_emb.weight", rows: file.writers.count, columns: width)
        contextTokens = try file.matrix("ctx_emb.weight", columns: width)
        contextPositions = try file.matrix("ctx_pos_emb.weight", columns: width)
        guard contextTokens.rows >= (config.textEnd ? 258 : 257),
              contextPositions.rows >= config.maxTextLength + (config.textEnd ? 1 : 0) else {
            throw PenGPTError.invalidModel("text context dimensions")
        }
        penProjection = config.bands == 0 ? nil : try file.matrix(
            "pen_pos_proj.weight", rows: width, columns: 4 * config.bands
        )
        let vectorSize = config.vector ?? 0
        self.vectorSize = vectorSize
        if vectorSize > 0 {
            vectorProjection = try file.matrix("ctx_proj.weight", rows: 4 * width, columns: vectorSize)
            vectorSlot = try file.matrix("ctx_slot.weight", rows: 1, columns: 4 * width).row(0)
            hintProjection = hasHints ? try file.matrix("hint_proj.weight", rows: width, columns: 40) : nil
        } else {
            vectorProjection = nil
            vectorSlot = []
            hintProjection = nil
        }
        head = try file.matrix("head.weight", rows: stroke.paths.count, columns: width)
        let residual = try file.vector("resid_scale", config.nLayer)
        let original = try file.vector("x0_scale", config.nLayer)
        layers = try (0 ..< config.nLayer).map { index in
            let prefix = "blocks.\(index)"
            let inner = try file.matrix(prefix + ".fc.weight", columns: width)
            return try DecoderLayer(
                selfAttention: attention(prefix + ".attn"),
                crossAttention: attention(prefix + ".cross_attn"),
                feedForward: inner,
                feedForwardOutput: file.matrix(prefix + ".fc_out.weight", rows: width, columns: inner.rows),
                residualScale: residual[index],
                embeddingScale: original[index]
            )
        }
        positionFrequencies = (0 ..< config.bands).map { Float(Double.pi / pow(2, Double($0 + 1))) }
        if usesPenRotary {
            let seq = headWidth / 4
            let pen = (headWidth / 2 - seq) / 2
            let seqValues = (0 ..< seq).map { Float(pow(10_000, -Double($0) / Double(seq))) }
            let penValues = (0 ..< pen).map { Float(2 * Double.pi / (4 * pow(128, Double($0) / Double(max(pen - 1, 1))))) }
            rotaryFrequencies = seqValues + penValues + penValues
        } else {
            rotaryFrequencies = (0 ..< headWidth / 2).map { Float(pow(10_000, -2 * Double($0) / Double(headWidth))) }
        }
    }

    struct Generation: Sendable {
        fileprivate var sequence: [Int]
        fileprivate let given: Int
        fileprivate let capacity: Int
        fileprivate var position = 0
        fileprivate var caches: [[KeyValues]]
        fileprivate var cross: [[KeyValues]] = []
        fileprivate var hints: [(Int, Int)] = []
        fileprivate var pen = SIMD2<Int>.zero
    }

    func makeGeneration(prefix: [Int] = [], maxTokens: Int?, guided: Bool) -> Generation {
        var sequence = [stroke.bos]
        if !prefix.isEmpty && prefix.count + 3 < maxSequence {
            sequence += prefix + [stroke.word, stroke.word]
        }
        let given = sequence.count
        let remaining = maxSequence - given
        let capacity = given + min(remaining, maxTokens ?? remaining)
        let caches = (0 ..< (guided ? 2 : 1)).map { _ in
            layers.map { _ in KeyValues(capacity: capacity, width: tokens.columns) }
        }
        return Generation(sequence: sequence, given: given, capacity: capacity, caches: caches)
    }

    func generate(_ text: String, writer: Int, prefix: [Int], options: HandwritingOptions,
                  rng: inout SamplingRandom, guidance: Float = 1,
                  vector: [Float]? = nil, ink: Double? = nil, strokeCount: Int? = nil,
                  style: [Float]? = nil) throws -> [Int] {
        var state = makeGeneration(prefix: prefix, maxTokens: options.maxTokensPerChunk, guided: guidance != 1)
        var output: [Int] = []
        while let token = try nextToken(&state, text: text, writer: writer, options: options, rng: &rng,
                                        guidance: guidance, vector: vector, ink: ink, strokeCount: strokeCount,
                                        style: style) {
            output.append(token)
        }
        return output
    }

    func isFinished(_ state: Generation) -> Bool {
        state.position >= state.capacity - 1 || state.sequence.last == stroke.end
    }

    func nextToken(_ state: inout Generation, text: String, writer: Int, options: HandwritingOptions,
                   rng: inout SamplingRandom, guidance: Float, vector: [Float]? = nil,
                   ink: Double? = nil, strokeCount: Int? = nil, style: [Float]? = nil) throws -> Int? {
        guard !isFinished(state) else { return nil }
        let width = tokens.columns
        let hints = hintDigits(ink: ink, strokeCount: strokeCount)
        if state.cross.isEmpty || !state.hints.elementsEqual(hints, by: ==) {
            let prompts = state.caches.count == 1 ? [text] : [text, ""]
            state.cross = prompts.enumerated().map { i, prompt in
                context(prompt, style: style ?? writers.row(writer), vector: i == 0 ? vector : nil, hints: hints)
            }
            state.hints = hints
        }
        while state.position < state.capacity - 1 {
            try Task.checkCancellation()
            let position = state.position
            state.position += 1
            let token = state.sequence[position]
            state.pen &+= stroke.deltas[token]
            var embedding = tokens.row(token)
            if let penProjection {
                let x = positionFrequencies.map { Float(state.pen.x) * $0 }
                let y = positionFrequencies.map { Float(state.pen.y) * $0 }
                add(penProjection.multiply(x.map(sin) + x.map(cos) + y.map(sin) + y.map(cos)), to: &embedding)
            }
            let x0 = normalize(embedding)
            let rotary = rotaryAngles(position, pen: state.pen)
            let cosine = rotary.map(cos)
            let sine = rotary.map(sin)

            var predictions: [[Float]] = []
            for branch in state.caches.indices {
                var x = x0
                for i in layers.indices {
                    let layer = layers[i]
                    for j in x.indices {
                        x[j] = layer.residualScale * x[j] + layer.embeddingScale * x0[j]
                    }
                    let attention = layer.selfAttention
                    let normalized = normalize(x)
                    let kv = attention.kv.multiply(normalized)
                    var q = normalizeHeads(attention.q.multiply(normalized), weights: attention.qNorm)
                    var k = normalizeHeads(Array(kv.prefix(width)), weights: attention.kNorm)
                    rotate(&q, cosine: cosine, sine: sine)
                    rotate(&k, cosine: cosine, sine: sine)
                    state.caches[branch][i].append(key: k, value: kv.suffix(width))
                    let selfOutput = attend(q, to: state.caches[branch][i])
                    add(attention.projection.multiply(selfOutput), to: &x)
                    let crossAttention = layer.crossAttention
                    let crossQ = normalizeHeads(crossAttention.q.multiply(normalize(x)), weights: crossAttention.qNorm)
                    add(crossAttention.projection.multiply(attend(crossQ, to: state.cross[branch][i])), to: &x)

                    let hidden = layer.feedForward.multiply(normalize(x)).map { value in
                        let positive = max(0, value)
                        return positive * positive
                    }
                    add(layer.feedForwardOutput.multiply(hidden), to: &x)
                }
                if position >= state.given - 1 {
                    predictions.append(head.multiply(normalize(x)).map { 15 * tanh($0 / 15) })
                }
            }
            if predictions.isEmpty { continue }
            var logits = predictions[0]
            if guidance != 1 {
                let conditional = logSoftmax(logits)
                let unconditional = logSoftmax(predictions[1])
                logits = zip(conditional, unconditional).map { $1 + guidance * ($0 - $1) }
            }
            let next = sample(logits, options: options, rng: &rng)
            state.sequence.append(next)
            return next
        }
        return nil
    }

    private func hintDigits(ink: Double?, strokeCount: Int?) -> [(Int, Int)] {
        guard hasHints else { return [] }
        let requested = [ink.map { Int(min(99, max(0, (2 * $0).rounded(.toNearestOrEven)))) },
                         strokeCount.map { min(99, max(0, $0)) }]
        return requested.enumerated().flatMap { index, value -> [(Int, Int)] in
            guard let value else { return [] }
            return [(2 * index, value / 10), (2 * index + 1, value % 10)]
        }
    }

    private func context(_ text: String, style: [Float], vector: [Float]?, hints: [(Int, Int)]) -> [KeyValues] {
        var context = [normalize(style)]
        let width = tokens.columns
        if let vectorProjection {
            if let vector, vector.contains(where: { $0 != 0 }) {
                var projection = vectorProjection.multiply(vector)
                add(vectorSlot, to: &projection)
                for offset in stride(from: 0, to: projection.count, by: width) {
                    context.append(normalize(Array(projection[offset ..< offset + width])))
                }
            }
            if let hintProjection, !hints.isEmpty {
                var flags = [Float](repeating: 0, count: 40)
                for (slot, digit) in hints { flags[slot * 10 + digit] = 1 }
                context.append(normalize(hintProjection.multiply(flags)))
            }
        } else {
            let textCount = maxTextLength - (hasHints ? 4 : 0)
            var ids = text.utf8.prefix(textCount).map { Int($0) + 1 }
            if textEnd { ids.append(257) }
            let positions = Array(ids.indices) + hints.map { textCount + (textEnd ? 1 : 0) + $0.0 }
            ids += hints.map { 258 + $0.0 * 10 + $0.1 }
            for (position, id) in zip(positions, ids) {
                var x = contextTokens.row(id)
                add(contextPositions.row(position), to: &x)
                context.append(normalize(x))
            }
        }
        return layers.map { layer in
            var cache = KeyValues(capacity: context.count, width: width)
            for x in context {
                let kv = layer.crossAttention.kv.multiply(x)
                let key = normalizeHeads(Array(kv.prefix(width)), weights: layer.crossAttention.kNorm)
                cache.append(key: key, value: kv.suffix(width))
            }
            return cache
        }
    }

    private func rotaryAngles(_ position: Int, pen: SIMD2<Int>) -> [Float] {
        guard penRotary else { return rotaryFrequencies.map { Float(position) * $0 } }
        let sequenceCount = rotaryFrequencies.count / 2
        let penCount = sequenceCount / 2
        return rotaryFrequencies.enumerated().map { i, frequency in
            let location = i < sequenceCount ? position : (i < sequenceCount + penCount ? pen.x : pen.y)
            return Float(location) * frequency
        }
    }

    private func rotate(_ x: inout [Float], cosine: [Float], sine: [Float]) {
        let half = cosine.count
        for start in stride(from: 0, to: x.count, by: 2 * half) {
            for i in 0 ..< half {
                let a = x[start + i]
                let b = x[start + half + i]
                x[start + i] = a * cosine[i] + b * sine[i]
                x[start + half + i] = b * cosine[i] - a * sine[i]
            }
        }
    }

    private func attend(_ query: [Float], to cache: KeyValues) -> [Float] {
        let width = query.count
        let headWidth = width / numHeads
        let rowCount = Int32(cache.count)
        let dimension = Int32(headWidth)
        let rowStride = Int32(width)
        let scale = 1 / sqrt(Float(headWidth))
        var scores = [Float](repeating: 0, count: cache.count)
        var output = [Float](repeating: 0, count: width)
        cache.keys.withUnsafeBufferPointer { keys in
            cache.values.withUnsafeBufferPointer { values in
                query.withUnsafeBufferPointer { q in
                    output.withUnsafeMutableBufferPointer { out in
                        for start in stride(from: 0, to: width, by: headWidth) {
                            cblas_sgemv(CblasRowMajor, CblasNoTrans, rowCount, dimension, scale,
                                        keys.baseAddress! + start, rowStride, q.baseAddress! + start, 1, 0, &scores, 1)
                            let total = exponentialWeights(&scores)
                            cblas_sgemv(CblasRowMajor, CblasTrans, rowCount, dimension, 1 / total,
                                        values.baseAddress! + start, rowStride, scores, 1, 0, out.baseAddress! + start, 1)
                        }
                    }
                }
            }
        }
        return output
    }
}

private struct KeyValues: Sendable {
    var keys: [Float]
    var values: [Float]
    var count = 0
    let width: Int

    init(capacity: Int, width: Int) {
        self.width = width
        keys = [Float](repeating: 0, count: capacity * width)
        values = keys
    }

    mutating func append(key: [Float], value: ArraySlice<Float>) {
        let start = count * width
        keys.replaceSubrange(start ..< start + width, with: key)
        values.replaceSubrange(start ..< start + width, with: value)
        count += 1
    }
}

func add(_ increment: [Float], to x: inout [Float]) {
    for i in x.indices { x[i] += increment[i] }
}

private func normalize(_ input: [Float]) -> [Float] {
    let factor = 1 / sqrt(cblas_sdot(Int32(input.count), input, 1, input, 1) / Float(input.count) + Float.ulpOfOne)
    return input.map { $0 * factor }
}

private func normalizeHeads(_ input: [Float], weights: [Float]) -> [Float] {
    var output = input
    let count = weights.count
    input.withUnsafeBufferPointer { buffer in
        for start in stride(from: 0, to: buffer.count, by: count) {
            let head = buffer.baseAddress! + start
            let meanSquare = cblas_sdot(Int32(count), head, 1, head, 1) / Float(count)
            let factor = 1 / sqrt(meanSquare + Float.ulpOfOne)
            for i in weights.indices { output[start + i] = head[i] * factor * weights[i] }
        }
    }
    return output
}

func exponentialWeights(_ logits: inout [Float]) -> Float {
    let maximum = logits.max()!
    var total: Float = 0
    for i in logits.indices {
        logits[i] = exp(logits[i] - maximum)
        total += logits[i]
    }
    return total
}

private func logSoftmax(_ logits: [Float]) -> [Float] {
    let maximum = logits.max()!
    let normalizer = log(logits.reduce(Float(0)) { $0 + exp($1 - maximum) })
    return logits.map { $0 - maximum - normalizer }
}

private func sample(_ logits: [Float], options: HandwritingOptions, rng: inout SamplingRandom) -> Int {
    let best = logits.indices.max { logits[$0] < logits[$1] }!
    guard options.temperature > 0 else { return best }
    let floor: Float
    if let topK = options.topK, topK < logits.count {
        floor = logits.sorted()[logits.count - topK]
    } else {
        floor = -.infinity
    }
    let peak = logits[best]
    let weights = logits.map { $0 >= floor ? exp((Double($0) - Double(peak)) / options.temperature) : 0 }
    var draw = rng.unit() * weights.reduce(0, +)
    for i in weights.indices {
        draw -= weights[i]
        if draw < 0 { return i }
    }
    return best
}

struct SamplingRandom {
    var state: UInt32

    // Match the browser runtime's seeded sampler.
    mutating func unit() -> Double {
        state &+= 0x6D2B79F5
        var value = (state ^ (state >> 15)) &* (state | 1)
        value ^= value &+ ((value ^ (value >> 7)) &* (value | 61))
        return Double(value ^ (value >> 14)) / 4_294_967_296
    }
}
