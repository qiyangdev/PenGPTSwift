import Foundation
import XCTest
@testable import PenGPT
import PenGPTModelImport

final class HandwritingTests: XCTestCase {
    private struct GreedyCase: Decodable {
        let text: String
        let writer: String
        let guidance: Float
        let tokens: [Int]
    }

    func testConvertedFixtureProducesUpstreamTokensAndCurves() async throws {
        let input = try fixture("tiny.model", "npz")
        let json = try Data(contentsOf: fixture("tiny.model", "json"))
        let cases = try JSONDecoder().decode([GreedyCase].self, from: json)
        let output = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".pengpt")
        defer { try? FileManager.default.removeItem(at: output) }

        try NumPyModel.convert(input, to: output)
        let model = try await HandwritingModel.load(weightsAt: output)
        XCTAssertEqual(model.writerIDs, ["", "ann", "bo"])

        var random = SamplingRandom(state: 0)
        for reference in cases {
            let writer = try XCTUnwrap(model.writerIDs.firstIndex(of: reference.writer))
            let tokens = try model.decoder.generate(reference.text, writer: writer, prefix: [],
                                                    options: .init(), rng: &random, guidance: reference.guidance)
            XCTAssertEqual(tokens, reference.tokens, reference.text)
        }

        // Golden curve from the browser decoder, including smoothing and resampling.
        let paths = model.decoder.stroke.decode([8, 0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 0, 0, 9])
        let path = try XCTUnwrap(paths.first)
        XCTAssertEqual(paths.count, 1)
        let expected: [SIMD2<Double>] = [
            .init(0, 0), .init(0.010722890, 0.001839236), .init(0.021117920, 0.004933154),
            .init(0.029391529, 0.011844955), .init(0.033341337, 0.021754559),
            .init(0.029768119, 0.031855550), .init(0.022417808, 0.039827269),
            .init(0.0125, 0.04375), .init(0.002582192, 0.039827269),
            .init(-0.004768119, 0.031855550), .init(-0.008341338, 0.021754559),
            .init(-0.004391529, 0.011844955), .init(0.003882080, 0.004933154),
            .init(0.014277110, 0.001839236), .init(0.025, 0),
        ]
        XCTAssertEqual(path.count, expected.count)
        for (actual, expected) in zip(path, expected) {
            XCTAssertEqual(actual.x, expected.x, accuracy: 1e-6)
            XCTAssertEqual(actual.y, expected.y, accuracy: 1e-6)
        }
    }

    func testPublishedHandwritingModel() async throws {
        guard let url = Bundle.module.url(forResource: "released-hand", withExtension: "pengpt", subdirectory: "Fixtures") else {
            throw XCTSkip("Convert the release weights to Tests/PenGPTTests/Fixtures/released-hand.pengpt to run this test.")
        }
        let model = try await HandwritingModel.load(weightsAt: url)
        XCTAssertEqual(model.writerIDs.count, 1193)
        let writer = try XCTUnwrap(model.writerIDs.firstIndex(of: "sam"))
        var random = SamplingRandom(state: 0)
        let raw = try model.decoder.generate("Hello world", writer: writer, prefix: [], options: .init(), rng: &random)
        XCTAssertEqual(raw, [
            204, 8, 12, 56, 46, 9, 452, 19, 8, 323, 24, 39, 130, 32, 21, 97, 9, 302, 8, 6, 108, 30, 17,
            15, 89, 30, 68, 140, 5, 510, 23, 26, 93, 68, 140, 17, 56, 238, 27, 78, 0, 17, 15, 44, 18, 4,
            389, 9, 524, 524, 2, 8, 6, 93, 50, 15, 85, 11, 127, 12, 18, 21, 73, 44, 18, 4, 21, 11, 167,
            28, 23, 26, 209, 68, 140, 17, 56, 238, 27, 78, 12, 83, 238, 371, 95, 56, 41, 50, 9, 523,
        ])

        let text = "The quick brown fox"
        let options = HandwritingOptions(lineWidth: 1)
        let ink = try await model.write(text, writer: "sam", options: options)
        XCTAssertEqual(ink.strokes.map { $0.points.count }, [12, 17, 57, 96, 1, 82, 122, 99, 18])
        let starts = [0, 3, 6, 7].compactMap { index in ink.strokes.indices.contains(index) ? ink.strokes[index].points.first : nil }
        XCTAssertEqual(starts.count, 4)
        for (actual, expected) in zip(starts, [-0.18, 0.55, 1.12, 1.65]) {
            XCTAssertEqual(actual.x, 0, accuracy: 1e-6)
            XCTAssertEqual(actual.y, expected, accuracy: 1e-6)
        }

        let style = HandwritingStyle(embedding: model.decoder.writers.row(writer), temperature: 0)
        let styled = try await model.write(text, style: style, options: options)
        XCTAssertEqual(styled, ink)

        var iterator = try model.writeWords(text, writer: "sam", options: options).makeAsyncIterator()
        let first = try await iterator.next()
        XCTAssertEqual(first?.strokes.map { $0.points.count }, [12, 17, 57])
        var streamed = try XCTUnwrap(first).strokes
        var wordCount = 1
        while let word = try await iterator.next() {
            streamed.append(contentsOf: word.strokes)
            wordCount += 1
        }
        XCTAssertEqual(wordCount, 4)
        XCTAssertEqual(streamed, ink.strokes)
    }

    func testWordStreamStopsAfterCancellation() async throws {
        let model = try HandwritingModel(file: NumPyModel.read(fixture("tiny.model", "npz")))
        let words = try model.writeWords("one two three", options: .init(maxTokensPerChunk: 1))
        try await Task {
            var iterator = words.makeAsyncIterator()
            let first = try await iterator.next()
            XCTAssertEqual(first?.strokes, [])

            withUnsafeCurrentTask { $0?.cancel() }
            do {
                _ = try await iterator.next()
                XCTFail("Expected cancellation before consuming buffered ink")
            } catch is CancellationError {
                let end = try await iterator.next()
                XCTAssertNil(end)
            }
        }.value
    }

    func testConvertsPhotoStyleVector() throws {
        let temp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temp) }
        let source = temp.appendingPathComponent("style.npy")
        let target = temp.appendingPathComponent("style.json")

        let header = Data("{'descr': '<f8', 'fortran_order': False, 'shape': (3,), }\n".utf8)
        var npy = Data([0x93, 78, 85, 77, 80, 89, 1, 0, UInt8(header.count), 0])
        npy += header
        for value in [0.25, -0.5, 0.4] {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { npy.append(contentsOf: $0) }
        }
        try npy.write(to: source)
        try NumPyModel.convertStyle(source, to: target)

        let converted = try HandwritingStyle.load(from: target)
        XCTAssertEqual(converted.embedding, [0.25, -0.5])
        XCTAssertEqual(converted.temperature, 0.4)
    }

    private func fixture(_ name: String, _ type: String) throws -> URL {
        try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: type, subdirectory: "Fixtures"))
    }
}

extension HandwritingTests: @unchecked Sendable {}
