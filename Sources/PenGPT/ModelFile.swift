import Foundation

public enum PenGPTError: LocalizedError, Sendable {
    case invalidModel(String)
    case unsupportedModel
    case unknownWriter(String)
    case invalidOptions

    public var errorDescription: String? {
        switch self {
        case .invalidModel(let reason): "Invalid PenGPT model: \(reason)"
        case .unsupportedModel: "This PenGPT model format or architecture is unsupported."
        case .unknownWriter(let id): "Unknown PenGPT writer: \(id)"
        case .invalidOptions: "Invalid handwriting options."
        }
    }
}

package struct ModelConfiguration: Decodable {
    let format: Int
    let nLayer: Int
    let nHead: Int
    let bands: Int
    let blockSize: Int
    let grid: Double
    let maxTextLength: Int
    let maxSeqLength: Int
    let textEnd: Bool
    let penRotary: Bool?
    let vector: Int?
    let hint: Double?
    let labelDrop: Double?

    enum CodingKeys: String, CodingKey {
        case format, bands, grid, vector, hint
        case nLayer = "n_layer"
        case nHead = "n_head"
        case blockSize = "block_size"
        case maxTextLength = "max_text_length"
        case maxSeqLength = "max_seq_length"
        case textEnd = "text_end"
        case penRotary = "pen_rotary"
        case labelDrop = "label_drop"
    }
}

package struct ModelFile: Codable {
    package let version: Int
    package let metadata: Data
    package let writers: [String]
    package let merges: [Int]
    package let tensors: [String: Tensor]
    let tokenizer: Data?

    package struct Tensor: Codable {
        package let shape: [Int]
        package let values: Data
        package let rowScales: Data?

        package init(shape: [Int], values: Data, rowScales: Data?) {
            self.shape = shape
            self.values = values
            self.rowScales = rowScales
        }

        func floats(named name: String) throws -> [Float] {
            let count = try elementCount(shape)
            let result: [Float]
            if let rowScales {
                guard shape.count == 2, values.count == count,
                    rowScales.count.isMultiple(of: 4), rowScales.count / 4 == shape[0]
                else {
                    throw PenGPTError.invalidModel(name)
                }
                let scales = rowScales.littleEndianFloats()
                let columns = shape[1]
                result = values.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
                    Array(unsafeUninitializedCapacity: count) { out, initialized in
                        for (row, scale) in scales.enumerated() {
                            let start = row * columns
                            for index in start ..< start + columns {
                                out[index] = Float(Int8(bitPattern: bytes[index])) * scale
                            }
                        }
                        initialized = count
                    }
                }
            } else {
                guard values.count.isMultiple(of: 4), values.count / 4 == count else {
                    throw PenGPTError.invalidModel(name)
                }
                result = values.littleEndianFloats()
            }
            guard result.allSatisfy(\.isFinite) else { throw PenGPTError.invalidModel(name) }
            return result
        }
    }

    package init(metadata: Data, writers: [String], merges: [Int], tensors: [String: Tensor], tokenizer: Data? = nil) {
        version = 1
        self.metadata = metadata
        self.writers = writers
        self.merges = merges
        self.tensors = tensors
        self.tokenizer = tokenizer
    }

    func configuration() throws -> ModelConfiguration {
        guard version == 1 else { throw PenGPTError.unsupportedModel }
        let config = try JSONDecoder().decode(ModelConfiguration.self, from: metadata)
        guard (1 ... 64).contains(config.nLayer), (1 ... 128).contains(config.nHead),
            (0 ... 32).contains(config.bands), (2 ... 8192).contains(config.blockSize),
            (2 ... 8192).contains(config.maxSeqLength), (1 ... 8192).contains(config.maxTextLength),
            ((config.hint ?? 0) == 0 || config.maxTextLength >= 4),
            config.grid.isFinite, config.grid > 0, config.grid <= 1,
            !writers.isEmpty
        else {
            throw PenGPTError.invalidModel("model configuration")
        }
        return config
    }

    func matrix(_ name: String, rows: Int? = nil, columns: Int? = nil) throws -> Matrix {
        guard let tensor = tensors[name], tensor.shape.count == 2,
            tensor.shape.allSatisfy({ $0 > 0 && $0 <= Int(Int32.max) }),
            rows == nil || rows == tensor.shape[0], columns == nil || columns == tensor.shape[1]
        else {
            throw PenGPTError.invalidModel(name)
        }
        return Matrix(rows: tensor.shape[0], columns: tensor.shape[1], values: try tensor.floats(named: name))
    }

    func vector(_ name: String, _ count: Int) throws -> [Float] {
        guard let tensor = tensors[name], tensor.shape == [count] else {
            throw PenGPTError.invalidModel(name)
        }
        return try tensor.floats(named: name)
    }

    package func encoded() throws -> Data {
        let encoder = PropertyListEncoder()
        encoder.outputFormat = .binary
        return try encoder.encode(self)
    }

    static func load(from url: URL) throws -> Self {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        return try PropertyListDecoder().decode(Self.self, from: data)
    }
}

package func elementCount(_ shape: [Int]) throws -> Int {
    var count = 1
    for dimension in shape {
        let (product, overflow) = count.multipliedReportingOverflow(by: dimension)
        guard dimension >= 0, !overflow else { throw PenGPTError.invalidModel("tensor dimensions") }
        count = product
    }
    return count
}

extension Data {
    func littleEndianFloats() -> [Float] {
        withUnsafeBytes { bytes in
            (0 ..< bytes.count / 4).map { index in
                let bits = bytes.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)
                return Float(bitPattern: UInt32(littleEndian: bits))
            }
        }
    }
}
