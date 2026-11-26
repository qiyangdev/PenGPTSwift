import Foundation
import XCTest
@testable import PenGPT

final class SketchTests: XCTestCase {
    func testReleasedSketchMatchesBrowser() async throws {
        guard let sketch = Bundle.module.url(forResource: "released-sketch", withExtension: "pengpt", subdirectory: "Fixtures"),
              let words = Bundle.module.url(forResource: "released-words", withExtension: "pengpt", subdirectory: "Fixtures") else {
            throw XCTSkip("Convert the sketch and words components to Tests/PenGPTTests/Fixtures/released-sketch.pengpt and released-words.pengpt.")
        }
        let encoder = try SketchTextEncoder(file: ModelFile.load(from: words))
        XCTAssertEqual(encoder.tokenIDs("Café's 2026 sign!"), [49406, 15304, 568, 273, 271, 273, 277, 2292, 256, 49407])
        let vector = try encoder.encode("a cat")
        let reference: [Float] = [0.01465451, -0.01048442, 0.00612729, 0.02284819,
                                  0.01161509, 0.02971007, 0.01173158, -0.55005050]
        for (actual, expected) in zip(vector, reference) {
            XCTAssertEqual(actual, expected, accuracy: 1e-5)
        }
        let file = try ModelFile.load(from: sketch)
        let config = try file.configuration()
        let strokes = try StrokeDecoder(grid: config.grid, merges: file.merges)
        let decoder = try Decoder(file: file, config: config, stroke: strokes)
        let source = try XCTUnwrap(file.writers.firstIndex(of: ""))
        var rng = SamplingRandom(state: 0)
        let raw = try decoder.generate("", writer: source, prefix: [], options: .init(maxTokensPerChunk: 48),
                                       rng: &rng, guidance: 2, vector: vector, ink: 8, strokeCount: 4)
        XCTAssertEqual(raw, [
            5, 299, 106, 8, 123, 6, 1, 180, 158, 29, 7, 45, 31, 221, 33, 54, 104, 80, 200, 1,
            49, 79, 146, 120, 146, 120, 28, 119, 11, 231, 56, 55, 107, 3, 39, 23, 95, 34, 92, 12,
            41, 169, 43, 26, 108, 36, 166, 13,
        ])
        let model = try await SketchModel.load(weightsAt: sketch, textEncoderAt: words)
        let settings = SketchOptions(temperature: 0, guidance: 2, ink: 8, strokeCount: 4, maxTokens: 48)
        let drawing = try await model.draw("a cat", options: settings)
        let first = try XCTUnwrap(drawing.strokes.first?.points.first)
        XCTAssertEqual(first.x, -0.3125, accuracy: 1e-6)
        XCTAssertEqual(first.y, -0.1625, accuracy: 1e-6)

        let session = try await model.startSketch("a cat", options: settings)
        let beginning = try await session.nextFrame(every: 8)
        let end = try await session.nextFrame(every: 40)
        XCTAssertEqual(beginning?.finished, false)
        XCTAssertEqual(end?.finished, true)
        XCTAssertEqual(end?.drawing, drawing)
        let exhausted = try await session.nextFrame()
        XCTAssertNil(exhausted)

        let live = try await model.startSketch("a cat", options: settings)
        let liveBeginning = try await live.nextFrame(every: 8)
        XCTAssertEqual(liveBeginning?.drawing, beginning?.drawing)
        let steered = try await live.nextFrame(every: 40, controls: .init(guidance: 1, ink: 35, strokeCount: 9))
        XCTAssertEqual(steered?.finished, true)
        XCTAssertNotEqual(steered?.drawing, drawing)
    }

    func testSVGContainsOnePathPerStroke() throws {
        let drawing = InkDrawing(strokes: [
            .init(points: [.init(x: -0.1, y: 0.05), .init(x: 0.2, y: 0.2)]),
            .init(points: [.init(x: 0.22, y: 0.18)]),
        ])
        let svg = try drawing.svg(ink: "#224466", background: nil)
        XCTAssertTrue(XMLParser(data: Data(svg.utf8)).parse())
        XCTAssertEqual(svg.components(separatedBy: "<path ").count - 1, 2)
        XCTAssertFalse(svg.contains("<rect"))
        XCTAssertTrue(svg.contains("stroke=\"#224466\""))
        XCTAssertThrowsError(try drawing.svg(ink: "invalid"))
        XCTAssertTrue(try InkDrawing(strokes: []).svg().contains("viewBox"))
    }
}

extension SketchTests: @unchecked Sendable {}
