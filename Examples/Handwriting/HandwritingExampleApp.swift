import Foundation
import PenGPT
import SwiftUI
import UniformTypeIdentifiers

@main
struct HandwritingExampleApp: App {
    var body: some Scene {
        WindowGroup {
            HandwritingScreen()
        }
    }
}

private struct HandwritingScreen: View {
    private static let sample = "The quick brown fox jumps over the lazy dog."

    private enum Mode {
        case handwriting, sketch
    }

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @FocusState private var inputFocused: Bool
    @State private var text = Self.sample
    @State private var prompt = "a cat sitting on a windowsill"
    @State private var mode = Mode.handwriting
    @State private var request: Request? = Request(text: Self.sample, mode: .handwriting)
    @State private var model: HandwritingModel?
    @State private var sketchModel: SketchModel?
    @State private var guidance = 2.0
    @State private var ink = 0.0
    @State private var strokes: [[InkPoint]] = []
    @State private var extent = CGRect.zero
    @State private var canvasWidth: CGFloat = 0
    @State private var handwritingLineWidth = 1.0
    @State private var error: String?
    @State private var paused = false

    var body: some View {
        NavigationStack {
            List {
                TextField(mode == .handwriting ? "Text to write" : "What to draw",
                          text: mode == .handwriting ? $text : $prompt, axis: .vertical)
                    .lineLimit(2 ... 5)
                    .textFieldStyle(.plain)
                    .focused($inputFocused)
                    .accessibilityIdentifier("handwriting-input")

                if let error {
                    Text(error)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                InkCanvas(
                    strokes: strokes,
                    extent: extent,
                    lineWidth: mode == .handwriting ? handwritingLineWidth : 1.2,
                    isSketch: mode == .sketch
                )
                .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { width in
                    if width > 0 { canvasWidth = width }
                }
                .listRowInsets(EdgeInsets())
                .listRowBackground(Color.clear)
                .listRowSeparator(.hidden)

                if mode == .sketch {
                    SketchTuning(guidance: $guidance, ink: $ink)
                }
            }
            .listStyle(.plain)
            .navigationTitle("PenGPT Handwriting")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Picker("Mode", selection: $mode) {
                        Text("Write").tag(Mode.handwriting)
                        Text("Draw").tag(Mode.sketch)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 160)
                    .disabled(request != nil)
                }
                ToolbarItemGroup(placement: .primaryAction) {
                    if request != nil {
                        Button {
                            paused.toggle()
                        } label: {
                            if paused {
                                Label("Resume", systemImage: "play.fill")
                            } else {
                                Label("Pause", systemImage: "pause.fill")
                            }
                        }
                        .accessibilityIdentifier("handwriting-pause")
                        Button("Stop", systemImage: "stop.fill") {
                            paused = false
                            request = nil
                        }
                            .accessibilityIdentifier("handwriting-stop")
                    } else {
                        if !strokes.isEmpty {
                            ShareLink(
                                item: SVGShare(drawing: InkDrawing(strokes: strokes.map(InkStroke.init(points:)))),
                                preview: SharePreview("PenGPT.svg")
                            )
                            .accessibilityIdentifier("handwriting-share")
                        }
                        Button(mode == .handwriting ? "Write" : "Draw", systemImage: "pencil", action: start)
                            .disabled(input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                            .accessibilityIdentifier("handwriting-write")
                    }
                }
            }
        }
        .onChange(of: mode) { _ in clearCanvas() }
        .task(id: runnableRequestID) { [request, canvasWidth] in
            guard let request else { return }
            if request.mode == .handwriting {
                guard canvasWidth > 0 else { return }
                handwritingLineWidth = InkCanvas.handwritingLineWidth(for: canvasWidth)
            }
            await run(request)
        }
    }

    private var input: String { mode == .handwriting ? text : prompt }

    private var runnableRequestID: UUID? {
        guard let request, request.mode == .sketch || canvasWidth > 0 else { return nil }
        return request.id
    }

    private func clearCanvas() {
        strokes = []
        extent = .zero
        error = nil
    }

    private func start() {
        inputFocused = false
        paused = false
        request = Request(text: input, mode: mode)
    }

    private func run(_ job: Request) async {
        guard request?.id == job.id, !Task.isCancelled else { return }
        strokes = []
        extent = .zero
        error = nil

        do {
            switch job.mode {
            case .handwriting: try await write(job.text)
            case .sketch: try await draw(job.text)
            }
        } catch is CancellationError {
            return
        } catch {
            guard request?.id == job.id, !Task.isCancelled else { return }
            self.error = error.localizedDescription
        }

        if request?.id == job.id { request = nil }
    }

    private func write(_ text: String) async throws {
        let handwriting = try await loadModel()
        var words = try handwriting.writeWords(
            text, writer: "sam", options: .init(lineWidth: handwritingLineWidth)
        ).makeAsyncIterator()
        while true {
            try await waitWhilePaused()
            guard let word = try await words.next() else { return }
            try await waitWhilePaused()
            for stroke in word.strokes {
                extendCanvas(with: [stroke.points])
                try await reveal(stroke.points)
            }
        }
    }

    private func draw(_ prompt: String) async throws {
        let sketch = try await loadSketch()
        let session = try await sketch.startSketch(prompt)
        while true {
            try await waitWhilePaused()
            let controls = SketchControls(guidance: Float(guidance), ink: ink > 0 ? ink : nil)
            guard let frame = try await session.nextFrame(controls: controls) else { return }
            try await waitWhilePaused()
            strokes = frame.drawing.strokes.map(\.points)
            if frame.finished {
                if strokes.isEmpty { error = String(localized: "No sketch was generated.") }
                return
            }
            try await Task.sleep(for: .milliseconds(16))
        }
    }

    private func extendCanvas(with paths: [[InkPoint]]) {
        extent = paths.lazy.flatMap { $0 }.reduce(extent) { bounds, point in
            bounds.union(CGRect(x: point.x, y: point.y, width: 0, height: 0))
        }
    }

    private func loadModel() async throws -> HandwritingModel {
        if let model { return model }
        guard let url = Bundle.main.url(forResource: "hand", withExtension: "pengpt", subdirectory: "Models") else {
            throw MissingWeights(names: "hand.pengpt")
        }
        let loaded = try await HandwritingModel.load(weightsAt: url)
        try Task.checkCancellation()
        model = loaded
        return loaded
    }

    private func loadSketch() async throws -> SketchModel {
        if let sketchModel { return sketchModel }
        guard let sketch = Bundle.main.url(forResource: "sketch", withExtension: "pengpt", subdirectory: "Models"),
              let words = Bundle.main.url(forResource: "words", withExtension: "pengpt", subdirectory: "Models") else {
            throw MissingWeights(names: "sketch.pengpt and words.pengpt")
        }
        let loaded = try await SketchModel.load(weightsAt: sketch, textEncoderAt: words)
        try Task.checkCancellation()
        sketchModel = loaded
        return loaded
    }

    private func reveal(_ points: [InkPoint]) async throws {
        try await waitWhilePaused()
        if reduceMotion || points.count <= 1 {
            strokes.append(points)
            return
        }

        let index = strokes.count
        strokes.append([])
        var start = 0
        while start < points.count {
            try await waitWhilePaused()
            let end = min(start + 5, points.count)
            strokes[index].append(contentsOf: points[start ..< end])
            start = end
            try await Task.sleep(for: .milliseconds(16))
        }
    }

    private func waitWhilePaused() async throws {
        while paused {
            try await Task.sleep(for: .milliseconds(32))
        }
        try Task.checkCancellation()
    }

    private struct Request {
        let id = UUID()
        let text: String
        let mode: Mode
    }

    private struct MissingWeights: LocalizedError {
        let names: String

        var errorDescription: String? {
            String(localized: "Add \(names) to the example’s Models folder, then rebuild.")
        }
    }

    private struct SVGShare: Transferable {
        let drawing: InkDrawing

        static var transferRepresentation: some TransferRepresentation {
            DataRepresentation(exportedContentType: .svg) { item in
                Data(try item.drawing.svg().utf8)
            }
            .suggestedFileName("PenGPT.svg")
        }
    }
}

private struct SketchTuning: View {
    @Binding var guidance: Double
    @Binding var ink: Double

    var body: some View {
        VStack(alignment: .leading) {
            Text("Guidance")
            Slider(value: $guidance, in: 1 ... 4) { Text("Guidance") }
            Text("Ink")
            Slider(value: $ink, in: 0 ... 49.5) { Text("Ink") }
        }
        .font(.footnote)
    }
}
