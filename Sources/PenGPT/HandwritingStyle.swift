import Foundation

/// A writer embedding and suggested sampling temperature, including styles learned from photographs.
public struct HandwritingStyle: Codable, Sendable {
    public let embedding: [Float]
    public let temperature: Double

    public init(embedding: [Float], temperature: Double) {
        self.embedding = embedding
        self.temperature = temperature
    }

    public static func load(from url: URL) throws -> Self {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Self.self, from: data)
    }
}
