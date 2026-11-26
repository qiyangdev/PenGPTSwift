// Swift adaptation of the PenGPT browser weight reader by Sam Greydanus and contributors.
// See NOTICE and LICENSE.

import Compression
import Foundation
import PenGPT

package enum NumPyModel {
    package enum Component: String {
        case hand, sketch, words
    }

    package static func convert(_ input: URL, to output: URL, component: Component? = nil) throws {
        let file = try read(input, component: component)
        let metadata = try JSONSerialization.jsonObject(with: file.metadata) as? [String: Any]
        if metadata?["bos"] != nil {
            try SketchModel.validate(file, isTextEncoder: true)
        } else if metadata?["format"] as? Int == 2 {
            try SketchModel.validate(file, isTextEncoder: false)
        } else {
            _ = try HandwritingModel(file: file)
        }
        try file.encoded().write(to: output, options: .atomic)
    }

    package static func convertStyle(_ url: URL, to output: URL) throws {
        let array = try NPY(Data(contentsOf: url))
        let width: Int
        switch array.type {
        case "<f4": width = 4
        case "<f8": width = 8
        default: throw PenGPTError.invalidModel("style vector data type")
        }
        guard array.shape.count == 1, let count = array.shape.first, (2 ... 8193).contains(count),
              array.values.count == count * width else {
            throw PenGPTError.invalidModel("style vector dimensions")
        }
        let values = try (0 ..< count).map { i -> Double in
            if width == 4 {
                return Double(Float(bitPattern: try array.values.integer(at: i * width, as: UInt32.self)))
            }
            return Double(bitPattern: try array.values.integer(at: i * width, as: UInt64.self))
        }
        let style = HandwritingStyle(embedding: values.dropLast().map(Float.init), temperature: values[count - 1])
        guard style.embedding.allSatisfy(\.isFinite), style.temperature.isFinite, style.temperature >= 0 else {
            throw PenGPTError.invalidModel("non-finite or invalid style vector")
        }
        try JSONEncoder().encode(style).write(to: output, options: .atomic)
    }

    package static func read(_ url: URL, component: Component? = nil) throws -> ModelFile {
        let archive = try Data(contentsOf: url, options: .mappedIfSafe)
        let arrays = try entries(in: archive, component: component ?? .hand)
        guard let meta = try arrays["meta"]?.strings().first else {
            throw PenGPTError.invalidModel("missing metadata")
        }
        let tokenizer = arrays["tokenizer"]
        if let tokenizer {
            guard tokenizer.type == "|u1", tokenizer.shape == [tokenizer.values.count] else {
                throw PenGPTError.invalidModel("invalid text tokenizer")
            }
        }

        let writers: [String]
        let mergeIDs: [Int]
        if tokenizer != nil {
            writers = []
            mergeIDs = []
        } else {
            guard let names = try arrays["writers"]?.strings(),
                  let merges = arrays["merges"], merges.type == "<i2",
                  merges.shape.count == 2, merges.shape.last == 3,
                  merges.values.count.isMultiple(of: 2),
                  merges.values.count / 2 == (try elementCount(merges.shape)) else {
                throw PenGPTError.invalidModel("missing or invalid writers or merges")
            }
            writers = names
            mergeIDs = try (0 ..< merges.values.count / 2).map {
                Int(Int16(bitPattern: try merges.values.integer(at: $0 * 2, as: UInt16.self)))
            }
        }
        var tensors: [String: ModelFile.Tensor] = [:]
        for (name, array) in arrays where !["meta", "writers", "merges", "tokenizer"].contains(name) && !name.hasSuffix(".scale") {
            let scales = arrays[name + ".scale"]
            switch (array.type, scales) {
            case ("<f4", nil):
                break
            case ("|i1", .some(let scale)) where array.shape.count == 2 &&
                scale.shape == [array.shape[0], 1] && scale.type == "<f4":
                break
            default:
                throw PenGPTError.invalidModel("unsupported tensor \(name)")
            }
            tensors[name] = .init(shape: array.shape, values: array.values, rowScales: scales?.values)
        }
        return ModelFile(metadata: Data(meta.utf8), writers: writers, merges: mergeIDs, tensors: tensors, tokenizer: tokenizer?.values)
    }

    private static func entries(in zip: Data, component: Component) throws -> [String: NPY] {
        let signature: UInt32 = 0x06054b50
        guard zip.count >= 22 else { throw PenGPTError.invalidModel("truncated ZIP") }
        var end = zip.count - 22
        let earliestEnd = max(0, end - Int(UInt16.max))
        while end >= earliestEnd {
            if try zip.integer(at: end, as: UInt32.self) == signature,
               try end + 22 + Int(zip.integer(at: end + 20, as: UInt16.self)) == zip.count { break }
            end -= 1
        }
        guard end >= earliestEnd,
              try zip.integer(at: end + 4, as: UInt32.self) == 0 else {
            throw PenGPTError.invalidModel("ZIP directory")
        }
        let count = Int(try zip.integer(at: end + 10, as: UInt16.self))
        var position = Int(try zip.integer(at: end + 16, as: UInt32.self))
        var arrays: [String: NPY] = [:]
        var expandedBytes = 0
        for _ in 0 ..< count {
            guard try zip.integer(at: position, as: UInt32.self) == 0x02014b50,
                  try zip.integer(at: position + 8, as: UInt16.self) & 1 == 0 else {
                throw PenGPTError.invalidModel("ZIP directory entry")
            }
            let method = try zip.integer(at: position + 10, as: UInt16.self)
            let size = Int(try zip.integer(at: position + 20, as: UInt32.self))
            let expanded = Int(try zip.integer(at: position + 24, as: UInt32.self))
            let nameLength = Int(try zip.integer(at: position + 28, as: UInt16.self))
            let extra = Int(try zip.integer(at: position + 30, as: UInt16.self))
            let comment = Int(try zip.integer(at: position + 32, as: UInt16.self))
            let local = Int(try zip.integer(at: position + 42, as: UInt32.self))
            let filename = try zip.bytes(at: position + 46, count: nameLength)
            position += 46 + nameLength + extra + comment
            guard let entryName = String(data: filename, encoding: .utf8), entryName.hasSuffix(".npy") else {
                throw PenGPTError.invalidModel("unsupported ZIP entry")
            }
            let parts = entryName.split(separator: "/", omittingEmptySubsequences: false)
            if parts.count == 2 && parts[0] != component.rawValue { continue }
            guard parts.count <= 2, let name = parts.last, !name.isEmpty,
                  try zip.integer(at: local, as: UInt32.self) == 0x04034b50 else {
                throw PenGPTError.invalidModel("unsupported ZIP entry")
            }
            expandedBytes += expanded
            guard expandedBytes <= 128 * 1024 * 1024, expanded > 0 else {
                throw PenGPTError.invalidModel("ZIP expansion exceeds the supported size")
            }
            let offset = try local + 30 + Int(zip.integer(at: local + 26, as: UInt16.self))
                + Int(zip.integer(at: local + 28, as: UInt16.self))
            var body = try zip.bytes(at: offset, count: size)
            if method == 8, !body.isEmpty {
                var decoded = Data(count: expanded + 1)
                let actual = decoded.withUnsafeMutableBytes { destination in
                    body.withUnsafeBytes { source in
                        compression_decode_buffer(
                            destination.bindMemory(to: UInt8.self).baseAddress!, destination.count,
                            source.bindMemory(to: UInt8.self).baseAddress!, source.count, nil, COMPRESSION_ZLIB
                        )
                    }
                }
                guard actual == expanded else { throw PenGPTError.invalidModel("ZIP decompression") }
                decoded.removeLast()
                body = decoded
            } else if method != 0 || body.count != expanded {
                throw PenGPTError.invalidModel("unsupported ZIP compression")
            }
            let key = String(name.dropLast(4))
            guard arrays[key] == nil else { throw PenGPTError.invalidModel("duplicate tensor \(key)") }
            arrays[key] = try NPY(body)
        }
        return arrays
    }
}

private struct NPY {
    let shape: [Int]
    let type: String
    let values: Data

    init(_ data: Data) throws {
        guard try data.bytes(at: 0, count: 6).elementsEqual([0x93, 78, 85, 77, 80, 89]) else {
            throw PenGPTError.invalidModel("NPY signature")
        }
        let version = try data.integer(at: 6, as: UInt8.self)
        let offset: Int
        let size: Int
        switch version {
        case 1:
            offset = 10
            size = Int(try data.integer(at: 8, as: UInt16.self))
        case 2, 3:
            offset = 12
            size = Int(try data.integer(at: 8, as: UInt32.self))
        default:
            throw PenGPTError.invalidModel("NPY version")
        }
        let header = String(decoding: try data.bytes(at: offset, count: size), as: UTF8.self)
        guard let descriptor = header.firstMatch(of: /'descr':\s*'([^']+)'/),
              let order = header.firstMatch(of: /'fortran_order':\s*(False|True)/), order.1 == "False",
              let dimensions = header.firstMatch(of: /'shape':\s*\(([^)]*)\)/) else {
            throw PenGPTError.invalidModel("unsupported NPY header")
        }
        type = String(descriptor.1)
        shape = try dimensions.1.split(separator: ",").compactMap { part in
            let number = part.trimmingCharacters(in: .whitespaces)
            guard !number.isEmpty else { return nil }
            guard let dimension = Int(number), dimension >= 0 else {
                throw PenGPTError.invalidModel("NPY shape")
            }
            return dimension
        }
        values = Data(data.dropFirst(offset + size))
    }

    func strings() throws -> [String] {
        guard type.hasPrefix("<U"), let width = Int(type.dropFirst(2)), width > 0,
              width <= values.count / 4, values.count.isMultiple(of: width * 4),
              values.count / (width * 4) == (try elementCount(shape)) else {
            throw PenGPTError.invalidModel("NPY Unicode array")
        }
        return try (0 ..< values.count / (4 * width)).map { index in
            let bytes = try values.bytes(at: index * width * 4, count: width * 4)
            guard let string = String(data: bytes, encoding: .utf32LittleEndian) else {
                throw PenGPTError.invalidModel("NPY Unicode scalar")
            }
            return String(string.prefix(while: { $0 != "\0" }))
        }
    }
}

private extension Data {
    func integer<T: FixedWidthInteger>(at offset: Int, as type: T.Type) throws -> T {
        guard offset >= 0, offset <= count - MemoryLayout<T>.size else {
            throw PenGPTError.invalidModel("truncated archive")
        }
        return withUnsafeBytes { T(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: type)) }
    }

    func bytes(at offset: Int, count length: Int) throws -> Data {
        guard offset >= 0, length >= 0, offset <= count - length else {
            throw PenGPTError.invalidModel("truncated archive")
        }
        return subdata(in: offset ..< offset + length)
    }
}
