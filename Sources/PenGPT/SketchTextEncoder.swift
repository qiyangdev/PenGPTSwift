// Swift adaptation of the PenGPT browser TinyCLIP text encoder by Sam Greydanus and contributors.
// See NOTICE and LICENSE.

import Accelerate
import Foundation

struct SketchTextEncoder: Sendable {
    private struct Configuration: Decodable {
        let width, nLayer, nHead, positions, bos, eos, vector: Int
        let eps: Float

        enum CodingKeys: String, CodingKey {
            case width, positions, bos, eos, vector, eps
            case nLayer = "n_layer"
            case nHead = "n_head"
        }
    }

    private struct Tokenizer: Decodable {
        let vocab: [String]
        let merges: [String]
    }

    private struct EmbeddingTable: Sendable {
        let width: Int
        let values: Data
        let scales: [Float]

        init(file: ModelFile, rows: Int, columns: Int) throws {
            let name = "embeddings.token_embedding.weight"
            guard let tensor = file.tensors[name], tensor.shape == [rows, columns],
                  let scaleData = tensor.rowScales, tensor.values.count == (try elementCount(tensor.shape)),
                  scaleData.count.isMultiple(of: 4), scaleData.count / 4 == rows else {
                throw PenGPTError.invalidModel(name)
            }
            width = columns
            values = tensor.values
            scales = scaleData.littleEndianFloats()
            guard scales.allSatisfy(\.isFinite) else { throw PenGPTError.invalidModel(name) }
        }

        func row(_ index: Int) -> [Float] {
            let scale = scales[index]
            let start = index * width
            return values.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
                (start ..< start + width).map { Float(Int8(bitPattern: bytes[$0])) * scale }
            }
        }
    }

    private struct Linear: Sendable {
        let weight: Matrix
        let bias: [Float]

        init(_ file: ModelFile, _ name: String, rows: Int? = nil, columns: Int) throws {
            let weight = try file.matrix(name + ".weight", rows: rows, columns: columns)
            self.weight = weight
            bias = try file.vector(name + ".bias", weight.rows)
        }

        func apply(_ x: [Float]) -> [Float] {
            var out = weight.multiply(x)
            add(bias, to: &out)
            return out
        }
    }

    private struct Norm: Sendable {
        let weight: [Float]
        let bias: [Float]

        init(_ file: ModelFile, _ name: String, _ width: Int) throws {
            weight = try file.vector(name + ".weight", width)
            bias = try file.vector(name + ".bias", width)
        }

        func apply(_ x: [Float], epsilon: Float) -> [Float] {
            let mean = x.reduce(0) { $0 + Double($1) } / Double(x.count)
            let variance = x.reduce(0) { sum, value in
                let delta = Double(value) - mean
                return sum + delta * delta
            } / Double(x.count)
            let scale = 1 / sqrt(variance + Double(epsilon))
            return x.indices.map { Float((Double(x[$0]) - mean) * scale * Double(weight[$0]) + Double(bias[$0])) }
        }
    }

    private struct Layer: Sendable {
        let attentionNorm: Norm
        let query, key, value, attentionOutput: Linear
        let mlpNorm: Norm
        let mlpInput, mlpOutput: Linear

        init(file: ModelFile, index: Int, width: Int) throws {
            let prefix = "encoder.layers.\(index)."
            attentionNorm = try Norm(file, prefix + "layer_norm1", width)
            query = try Linear(file, prefix + "self_attn.q_proj", rows: width, columns: width)
            key = try Linear(file, prefix + "self_attn.k_proj", rows: width, columns: width)
            value = try Linear(file, prefix + "self_attn.v_proj", rows: width, columns: width)
            attentionOutput = try Linear(file, prefix + "self_attn.out_proj", rows: width, columns: width)
            mlpNorm = try Norm(file, prefix + "layer_norm2", width)
            mlpInput = try Linear(file, prefix + "mlp.fc1", columns: width)
            mlpOutput = try Linear(file, prefix + "mlp.fc2", rows: width, columns: mlpInput.weight.rows)
        }
    }

    private static let tokenPattern = try! NSRegularExpression(
        pattern: #"'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+"#
    )
    private static let byteAlphabet: [String] = {
        var next = 256
        return (0 ..< 256).map { value in
            if (33 ... 126).contains(value) || (161 ... 172).contains(value) || value >= 174 {
                return String(UnicodeScalar(value)!)
            }
            defer { next += 1 }
            return String(UnicodeScalar(next)!)
        }
    }()

    private let config: Configuration
    private let ids: [String: Int]
    private let ranks: [String: Int]
    private let tokens: EmbeddingTable
    private let positions: Matrix
    private let layers: [Layer]
    private let finalNorm: Norm
    private let projection: Linear
    var vectorSize: Int { config.vector }

    init(file: ModelFile) throws {
        guard file.version == 1, let tokenizer = file.tokenizer else { throw PenGPTError.unsupportedModel }
        let config = try JSONDecoder().decode(Configuration.self, from: file.metadata)
        let vocabulary = try JSONDecoder().decode(Tokenizer.self, from: tokenizer)
        guard (1 ... 64).contains(config.nLayer), (1 ... 128).contains(config.nHead),
              (1 ... 4096).contains(config.width), config.width.isMultiple(of: config.nHead),
              (2 ... 1024).contains(config.positions), config.eps.isFinite, config.eps > 0,
              vocabulary.vocab.indices.contains(config.bos), vocabulary.vocab.indices.contains(config.eos) else {
            throw PenGPTError.invalidModel("sketch text encoder configuration")
        }
        self.config = config
        ids = Dictionary(vocabulary.vocab.enumerated().map { ($1, $0) }, uniquingKeysWith: { first, _ in first })
        ranks = Dictionary(vocabulary.merges.enumerated().map { ($1, $0) }, uniquingKeysWith: { first, _ in first })
        tokens = try EmbeddingTable(file: file, rows: vocabulary.vocab.count, columns: config.width)
        positions = try file.matrix("embeddings.position_embedding.weight", rows: config.positions, columns: config.width)
        layers = try (0 ..< config.nLayer).map { try Layer(file: file, index: $0, width: config.width) }
        finalNorm = try Norm(file, "final_layer_norm", config.width)
        projection = try Linear(file, "head", rows: config.vector, columns: config.width)
    }

    func tokenIDs(_ text: String) -> [Int] {
        let text = text.precomposedStringWithCanonicalMapping
            .split(whereSeparator: \.isWhitespace).joined(separator: " ").lowercased()
        let input = text as NSString
        var result = [config.bos]
        for match in Self.tokenPattern.matches(in: text, range: NSRange(location: 0, length: input.length)) {
            var symbols = input.substring(with: match.range).utf8.map { Self.byteAlphabet[Int($0)] }
            symbols[symbols.count - 1] += "</w>"
            while symbols.count > 1 {
                var best = Int.max
                var pair: (String, String)?
                for i in 0 ..< symbols.count - 1 {
                    let a = symbols[i]
                    let b = symbols[i + 1]
                    if let rank = ranks[a + " " + b], rank < best {
                        best = rank
                        pair = (a, b)
                    }
                }
                guard let (a, b) = pair else { break }
                var merged: [String] = []
                var index = 0
                while index < symbols.count {
                    if index + 1 < symbols.count && symbols[index] == a && symbols[index + 1] == b {
                        merged.append(a + b)
                        index += 2
                    } else {
                        merged.append(symbols[index])
                        index += 1
                    }
                }
                symbols = merged
            }
            result += symbols.map { ids[$0] ?? config.eos }.prefix(config.positions - 1 - result.count)
            if result.count == config.positions - 1 { break }
        }
        result.append(config.eos)
        return result
    }

    func encode(_ text: String) throws -> [Float] {
        var hidden = tokenIDs(text).enumerated().map { position, token in
            var row = tokens.row(token)
            add(positions.row(position), to: &row)
            return row
        }
        for layer in layers {
            try Task.checkCancellation()
            let states = hidden.map { layer.attentionNorm.apply($0, epsilon: config.eps) }
            let queries = states.map(layer.query.apply)
            let keys = states.map(layer.key.apply)
            let values = states.map(layer.value.apply)
            for position in hidden.indices {
                let attended = attend(at: position, query: queries[position], keys: keys, values: values)
                add(layer.attentionOutput.apply(attended), to: &hidden[position])
                let normalized = layer.mlpNorm.apply(hidden[position], epsilon: config.eps)
                let activated = layer.mlpInput.apply(normalized).map { $0 * 0.5 * (1 + erf($0 / sqrt(2))) }
                add(layer.mlpOutput.apply(activated), to: &hidden[position])
            }
        }
        let output = projection.apply(finalNorm.apply(hidden[hidden.count - 1], epsilon: config.eps))
        let norm = cblas_snrm2(Int32(output.count), output, 1)
        guard norm.isFinite, norm > 0 else { throw PenGPTError.invalidModel("sketch text embedding") }
        return output.map { $0 / norm }
    }

    private func attend(at position: Int, query: [Float], keys: [[Float]], values: [[Float]]) -> [Float] {
        let count = config.width / config.nHead
        let scale = 1 / sqrt(Float(count))
        var output = [Float](repeating: 0, count: config.width)
        for start in stride(from: 0, to: config.width, by: count) {
            var scores = (0 ... position).map { row in
                var dot: Float = 0
                for d in start ..< start + count { dot += query[d] * keys[row][d] }
                return dot * scale
            }
            let total = exponentialWeights(&scores)
            for (row, score) in scores.enumerated() {
                let probability = score / total
                for d in start ..< start + count { output[d] += probability * values[row][d] }
            }
        }
        return output
    }
}
