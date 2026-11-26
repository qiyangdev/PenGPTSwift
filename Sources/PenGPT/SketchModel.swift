import Foundation

public struct SketchOptions: Sendable {
    public var temperature: Double
    public var topK: Int?
    public var seed: UInt32
    public var guidance: Float?
    public var ink: Double?
    public var strokeCount: Int?
    public var maxTokens: Int?

    public init(temperature: Double = 0.6, topK: Int? = nil, seed: UInt32 = 0, guidance: Float? = nil,
                ink: Double? = nil, strokeCount: Int? = nil, maxTokens: Int? = nil) {
        self.temperature = temperature
        self.topK = topK
        self.seed = seed
        self.guidance = guidance
        self.ink = ink
        self.strokeCount = strokeCount
        self.maxTokens = maxTokens
    }
}

/// Changes to these controls affect tokens generated after the next frame begins.
public struct SketchControls: Sendable {
    public var guidance: Float?
    public var ink: Double?
    public var strokeCount: Int?

    public init(guidance: Float? = nil, ink: Double? = nil, strokeCount: Int? = nil) {
        self.guidance = guidance
        self.ink = ink
        self.strokeCount = strokeCount
    }

    fileprivate func validated(for decoder: Decoder) throws -> Float {
        let resolved = guidance ?? (decoder.labelDrop ? 2 : 1)
        guard resolved.isFinite, decoder.labelDrop || resolved == 1, ink?.isFinite ?? true else {
            throw PenGPTError.invalidOptions
        }
        return resolved
    }
}

public struct SketchFrame: Sendable {
    public let drawing: InkDrawing
    public let finished: Bool
}

/// Pulls a sketch from the same decoder state while preserving the sampling sequence.
public actor SketchSession {
    private let decoder: Decoder
    private let vector: [Float]?
    private let source: Int
    private let options: HandwritingOptions
    private var controls: SketchControls
    private var generation: Decoder.Generation
    private var random: SamplingRandom
    private var tokens: [Int] = []

    fileprivate init(decoder: Decoder, vector: [Float]?, source: Int, options: HandwritingOptions,
                     controls: SketchControls, live: Bool) {
        self.decoder = decoder
        self.vector = vector
        self.source = source
        self.options = options
        self.controls = controls
        generation = decoder.makeGeneration(maxTokens: options.maxTokensPerChunk, guided: live)
        random = SamplingRandom(state: options.seed)
    }

    /// Advances up to `every` tokens and returns a full snapshot for replacement rendering.
    /// Call again with updated controls to steer the remaining drawing.
    public func nextFrame(every: Int = 8, controls update: SketchControls? = nil) throws -> SketchFrame? {
        guard every > 0 else { throw PenGPTError.invalidOptions }
        guard !decoder.isFinished(generation) else { return nil }
        let controls = update ?? self.controls
        let guidance = try controls.validated(for: decoder)
        self.controls = controls

        for _ in 0 ..< every {
            guard let token = try decoder.nextToken(
                &generation, text: "", writer: source, options: options, rng: &random,
                guidance: guidance, vector: vector, ink: controls.ink, strokeCount: controls.strokeCount
            ) else { break }
            if token < decoder.stroke.pad { tokens.append(token) }
        }
        let paths = decoder.stroke.decode(tokens)
        let strokes = paths.map { InkStroke(points: $0.map { InkPoint(x: $0.x, y: $0.y) }) }
        return SketchFrame(drawing: InkDrawing(strokes: strokes), finished: decoder.isFinished(generation))
    }
}

/// The sketch decoder and the prompt encoder from the upstream browser demo.
public struct SketchModel: Sendable {
    public let sourceIDs: [String]
    private let decoder: Decoder
    private let encoder: SketchTextEncoder

    public static func load(weightsAt sketchURL: URL, textEncoderAt encoderURL: URL) async throws -> Self {
        try Task.checkCancellation()
        let decoderFile = try ModelFile.load(from: sketchURL)
        return try Self(decoderFile: decoderFile, encoderFile: ModelFile.load(from: encoderURL))
    }

    init(decoderFile: ModelFile, encoderFile: ModelFile) throws {
        decoder = try Self.makeDecoder(decoderFile)
        encoder = try SketchTextEncoder(file: encoderFile)
        guard decoder.vectorSize == encoder.vectorSize else {
            throw PenGPTError.invalidModel("sketch and text encoder dimensions differ")
        }
        sourceIDs = decoderFile.writers
    }

    package static func validate(_ file: ModelFile, isTextEncoder: Bool) throws {
        if isTextEncoder {
            _ = try SketchTextEncoder(file: file)
        } else {
            _ = try makeDecoder(file)
        }
    }

    private static func makeDecoder(_ file: ModelFile) throws -> Decoder {
        let config = try file.configuration()
        guard config.format == 2, let vector = config.vector, (1 ... 8192).contains(vector) else {
            throw PenGPTError.unsupportedModel
        }
        return try Decoder(file: file, config: config, stroke: StrokeDecoder(grid: config.grid, merges: file.merges))
    }

    /// Generates one drawing with coordinates in model units.
    public func draw(_ prompt: String, source: String = "", options: SketchOptions = .init()) async throws -> InkDrawing {
        let session = try makeSession(prompt, source: source, options: options, live: false)
        return try await session.nextFrame(every: Int.max)?.drawing ?? InkDrawing(strokes: [])
    }

    /// Creates a session that can stream progressive snapshots and accept live controls.
    public func startSketch(_ prompt: String, source: String = "", options: SketchOptions = .init()) async throws -> SketchSession {
        try makeSession(prompt, source: source, options: options, live: true)
    }

    private func makeSession(_ prompt: String, source: String, options: SketchOptions, live: Bool) throws -> SketchSession {
        let controls = SketchControls(guidance: options.guidance, ink: options.ink, strokeCount: options.strokeCount)
        let guidance = try controls.validated(for: decoder)
        guard options.temperature.isFinite, options.temperature >= 0,
              (options.topK ?? 1) > 0, (options.maxTokens ?? 1) > 0 else {
            throw PenGPTError.invalidOptions
        }
        guard let source = sourceIDs.firstIndex(of: source) else { throw PenGPTError.unknownWriter(source) }
        let vector = prompt.isEmpty ? nil : try encoder.encode(prompt)
        let sampling = HandwritingOptions(temperature: options.temperature, topK: options.topK,
                                          seed: options.seed, maxTokensPerChunk: options.maxTokens)
        return SketchSession(decoder: decoder, vector: vector, source: source, options: sampling,
                             controls: controls, live: guidance != 1 || (live && decoder.labelDrop))
    }
}
