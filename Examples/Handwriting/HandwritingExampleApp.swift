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
        #if os(macOS)
            .defaultSize(width: 900, height: 540)
        #endif
    }
}

private struct HandwritingScreen: View {
    private static let sample = "The quick brown fox jumps over the lazy dog."
    private static let defaultWriter = "sam"

    private enum Mode {
        case handwriting, sketch
    }

    #if os(iOS)
        private enum Sheet: Identifiable {
            case writer, sketchSettings

            var id: Self { self }
        }
    #endif

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @FocusState private var inputFocused: Bool
    @State private var text = Self.sample
    @State private var prompt = "a cat sitting on a windowsill"
    @State private var mode = Mode.handwriting
    @State private var writerID = Self.defaultWriter
    #if os(macOS)
        @State private var showsInspector = true
    #else
        @State private var activeSheet: Sheet?
    #endif
    @State private var request: Request? = Request(
        text: Self.sample,
        mode: .handwriting,
        writerID: Self.defaultWriter,
        sourceID: ""
    )
    @State private var model: HandwritingModel?
    @State private var sketchModel: SketchModel?
    @State private var sketchLoad: Task<SketchModel, Error>?
    @State private var sketchSourceID = ""
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
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if let error {
                        Text(error)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .padding(.horizontal)
                    }

                    InkCanvas(
                        strokes: strokes,
                        extent: extent,
                        lineWidth: mode == .handwriting
                            ? handwritingLineWidth : 1.2,
                        isSketch: mode == .sketch
                    )
                }
            }
            .onGeometryChange(for: CGFloat.self) {
                $0.size.width
            } action: { width in
                if width > 0 { canvasWidth = width }
            }
            .navigationTitle("PenGPT Handwriting")
            #if os(iOS)
                .scrollDismissesKeyboard(.interactively)
                .navigationBarTitleDisplayMode(.large)
            #endif
            .toolbar {
                #if os(iOS)
                    ToolbarItem(placement: .topBarLeading) {
                        modePicker.frame(width: 160)
                    }
                    ToolbarItem(placement: .bottomBar) {
                        inputBar
                            .frame(width: min(max(canvasWidth - 64, 180), 520))
                    }
                #endif
                ToolbarItemGroup(placement: .primaryAction) {
                    #if os(iOS)
                        if mode == .handwriting {
                            Button {
                                inputFocused = false
                                activeSheet = .writer
                            } label: {
                                Label(WriterRows.name(for: writerID), systemImage: "person.crop.circle")
                            }
                            .disabled(model == nil)
                            .accessibilityLabel("Choose Writer, \(WriterRows.name(for: writerID)) selected")
                            .accessibilityIdentifier("handwriting-writer")
                        } else {
                            Button("Drawing Settings", systemImage: "slider.horizontal.3") {
                                inputFocused = false
                                activeSheet = .sketchSettings
                            }
                            .accessibilityIdentifier("sketch-settings")
                        }
                    #endif

                    if request == nil && !strokes.isEmpty {
                        ShareLink(
                            item: SVGShare(drawing: InkDrawing(strokes: strokes.map(InkStroke.init(points:)))),
                            preview: SharePreview("PenGPT.svg")
                        )
                        .accessibilityIdentifier("handwriting-share")
                    }

                    #if os(macOS)
                        Button("Inspector", systemImage: "sidebar.right") {
                            showsInspector.toggle()
                        }
                        .accessibilityIdentifier("handwriting-inspector")
                    #endif
                }
            }
        }
        #if os(macOS)
            .inspector(isPresented: $showsInspector) {
                inspectorContent
                .inspectorColumnWidth(min: 240, ideal: 300, max: 440)
            }
            .onChange(of: mode) { clearCanvas() }
        #else
            .sheet(item: $activeSheet) { sheet in
                switch sheet {
                case .writer:
                    if let model {
                        WriterPicker(writerIDs: model.writerIDs, selected: writerID) { selected in
                            activeSheet = nil
                            selectWriter(selected)
                        }
                    }
                case .sketchSettings:
                    SketchSettings(fields: sketchSettingsFields)
                    .presentationDetents([.medium, .large])
                }
            }
            .onChange(of: mode) { _ in clearCanvas() }
        #endif
        .task(id: runnableRequestID) { [request, canvasWidth] in
            guard let request else { return }
            if request.mode == .handwriting {
                guard canvasWidth > 0 else { return }
                handwritingLineWidth = InkCanvas.handwritingLineWidth(
                    for: canvasWidth
                )
            }
            await run(request)
        }
    }

    private var modePicker: some View {
        Picker("Mode", selection: $mode) {
            Text("Write").tag(Mode.handwriting)
            Text("Draw").tag(Mode.sketch)
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .disabled(request != nil)
    }

    #if os(macOS)
        private var inspectorContent: some View {
            Form {
                Section {
                    modePicker
                }
                Section(mode == .handwriting ? "Text" : "Prompt") {
                    inputField
                    playbackButton
                }
                if mode == .handwriting {
                    if let model {
                        Section {
                            Picker("Writer", selection: Binding(get: { writerID }, set: selectWriter)) {
                                ForEach(model.writerIDs, id: \.self) { id in
                                    Text(WriterRows.name(for: id)).tag(id)
                                }
                            }
                            .pickerStyle(.menu)
                            .accessibilityIdentifier("handwriting-writer")
                        }
                    }
                } else {
                    sketchSettingsFields
                }
            }
            .formStyle(.grouped)
        }
    #else
        private var inputBar: some View {
            HStack(spacing: 0) {
                inputField.padding(.horizontal)
                playbackButton
            }
        }
    #endif

    private var sketchSettingsFields: SketchSettingsFields {
        SketchSettingsFields(sourceID: $sketchSourceID, guidance: $guidance, ink: $ink) {
            try await loadSketch().sourceIDs
        }
    }

    private var inputField: some View {
        #if os(macOS)
            let axis = Axis.vertical
        #else
            let axis = Axis.horizontal
        #endif
        return TextField(
            mode == .handwriting ? "Text to write" : "What to draw",
            text: mode == .handwriting ? $text : $prompt,
            axis: axis
        )
        #if os(macOS)
            .labelsHidden()
        #endif
        .frame(maxWidth: .infinity)
        .focused($inputFocused)
        .accessibilityIdentifier("handwriting-input")
    }

    private var playbackButton: some View {
        let icon = request != nil && !paused ? "pause.fill" : "play.fill"
        return Button {
            if request == nil {
                start()
            } else {
                paused.toggle()
            }
        } label: {
            #if os(macOS)
                Label(playbackLabel, systemImage: icon)
            #else
                Image(systemName: icon)
                    .frame(width: 44, height: 44)
                    .contentShape(Rectangle())
            #endif
        }
        .accessibilityLabel(playbackLabel)
        .buttonStyle(.borderless)
        .disabled(request == nil && !canStart)
        .accessibilityIdentifier(request == nil ? "handwriting-write" : "handwriting-pause")
    }

    private var input: String { mode == .handwriting ? text : prompt }

    private var playbackLabel: LocalizedStringKey {
        if request == nil { return mode == .handwriting ? "Write" : "Draw" }
        return paused ? "Resume" : "Pause"
    }

    private var canStart: Bool {
        !input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var runnableRequestID: UUID? {
        guard let request, request.mode == .sketch || canvasWidth > 0 else {
            return nil
        }
        return request.id
    }

    private func selectWriter(_ selected: String) {
        guard selected != writerID else { return }
        writerID = selected
        if canStart {
            start()
        } else {
            request = nil
            clearCanvas()
        }
    }

    private func clearCanvas() {
        strokes = []
        extent = .zero
        error = nil
    }

    private func start() {
        inputFocused = false
        paused = false
        request = Request(text: input, mode: mode, writerID: writerID, sourceID: sketchSourceID)
    }

    private func run(_ job: Request) async {
        guard request?.id == job.id, !Task.isCancelled else { return }
        strokes = []
        extent = .zero
        error = nil

        do {
            switch job.mode {
            case .handwriting: try await write(job.text, writerID: job.writerID)
            case .sketch: try await draw(job.text, sourceID: job.sourceID)
            }
        } catch is CancellationError {
            return
        } catch {
            guard request?.id == job.id, !Task.isCancelled else { return }
            self.error = error.localizedDescription
        }

        if request?.id == job.id { request = nil }
    }

    private func write(_ text: String, writerID: String) async throws {
        let handwriting = try await loadModel()
        var words = try handwriting.writeWords(
            text,
            writer: writerID,
            options: .init(lineWidth: handwritingLineWidth)
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

    private func draw(_ prompt: String, sourceID: String) async throws {
        let sketch = try await loadSketch()
        try Task.checkCancellation()
        let source = sketch.sourceIDs.contains(sourceID) ? sourceID : (sketch.sourceIDs.first ?? "")
        let session = try await sketch.startSketch(prompt, source: source)
        while true {
            try await waitWhilePaused()
            let controls = SketchControls(
                guidance: Float(guidance),
                ink: ink > 0 ? ink : nil
            )
            guard let frame = try await session.nextFrame(controls: controls)
            else { return }
            try await waitWhilePaused()
            strokes = frame.drawing.strokes.map(\.points)
            if frame.finished {
                if strokes.isEmpty {
                    error = String(localized: "No sketch was generated.")
                }
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
        guard
            let url = Bundle.main.url(
                forResource: "hand",
                withExtension: "pengpt",
                subdirectory: "Models"
            )
        else {
            throw MissingWeights(names: "hand.pengpt")
        }
        let loaded = try await HandwritingModel.load(weightsAt: url)
        try Task.checkCancellation()
        model = loaded
        return loaded
    }

    private func loadSketch() async throws -> SketchModel {
        if let sketchModel { return sketchModel }
        if let sketchLoad { return try await sketchLoad.value }
        guard
            let sketch = Bundle.main.url(
                forResource: "sketch",
                withExtension: "pengpt",
                subdirectory: "Models"
            ),
            let words = Bundle.main.url(
                forResource: "words",
                withExtension: "pengpt",
                subdirectory: "Models"
            )
        else {
            throw MissingWeights(names: "sketch.pengpt and words.pengpt")
        }
        let task = Task {
            try await SketchModel.load(weightsAt: sketch, textEncoderAt: words)
        }
        sketchLoad = task
        defer { sketchLoad = nil }

        let loaded = try await task.value
        if !loaded.sourceIDs.contains(sketchSourceID), let first = loaded.sourceIDs.first {
            sketchSourceID = first
        }
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
        let writerID: String
        let sourceID: String
    }

    private struct MissingWeights: LocalizedError {
        let names: String

        var errorDescription: String? {
            String(
                localized:
                    "Add \(names) to the example’s Models folder, then rebuild."
            )
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

#if os(iOS)
    private struct WriterPicker: View {
        let writerIDs: [String]
        let selected: String
        let select: (String) -> Void

        @Environment(\.dismiss) private var dismiss
        @State private var query = ""

        var body: some View {
            NavigationStack {
                WriterRows(writerIDs: writerIDs, selected: selected, query: query, select: select)
                    .searchable(
                        text: $query,
                        placement: .navigationBarDrawer(displayMode: .always),
                        prompt: "Search writers"
                    )
                    .navigationTitle("Choose Writer")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) {
                            Button("Cancel") { dismiss() }
                        }
                    }
            }
        }
    }
#endif

private struct WriterRows: View {
    let writerIDs: [String]
    let selected: String
    let query: String
    let select: (String) -> Void

    var body: some View {
        List(matchingWriters, id: \.self) { id in
            Button {
                select(id)
            } label: {
                HStack {
                    Text(Self.name(for: id))
                    Spacer()
                    if id == selected {
                        Image(systemName: "checkmark")
                            .foregroundStyle(Color.accentColor)
                            .accessibilityHidden(true)
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityAddTraits(id == selected ? .isSelected : [])
            .accessibilityIdentifier("writer-\(id)")
        }
    }

    private var matchingWriters: [String] {
        let term = query.trimmingCharacters(in: .whitespacesAndNewlines)
        return term.isEmpty
            ? writerIDs
            : writerIDs.filter {
                Self.name(for: $0).localizedStandardContains(term)
            }
    }

    static func name(for id: String) -> String {
        id.isEmpty ? String(localized: "Default") : id
    }
}

#if os(iOS)
    private struct SketchSettings: View {
        let fields: SketchSettingsFields

        @Environment(\.dismiss) private var dismiss

        var body: some View {
            NavigationStack {
                Form { fields }
                    .navigationTitle("Drawing Settings")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Done") { dismiss() }
                        }
                    }
            }
        }
    }
#endif

private struct SketchSettingsFields: View {
    @Binding var sourceID: String
    @Binding var guidance: Double
    @Binding var ink: Double
    let loadSources: () async throws -> [String]

    @State private var sourceIDs: [String] = []
    @State private var sourceError: String?

    var body: some View {
        Group {
            Section {
                if !sourceIDs.isEmpty {
                    Picker("Source", selection: $sourceID) {
                        ForEach(sourceIDs, id: \.self) { id in
                            if id.isEmpty {
                                Text("Default").tag(id)
                            } else {
                                Text(id).tag(id)
                            }
                        }
                    }
                    .pickerStyle(.menu)
                    .accessibilityIdentifier("sketch-source")
                } else if let sourceError {
                    Text(sourceError)
                        .foregroundStyle(.secondary)
                } else {
                    ProgressView("Loading sources…")
                }
            } footer: {
                Text("The source applies to the next drawing.")
            }

            Section("Guidance") {
                Slider(value: $guidance, in: 1 ... 4) { Text("Guidance") }
                    #if os(macOS)
                        .labelsHidden()
                    #endif
                    .accessibilityIdentifier("sketch-guidance")
            }
            Section("Ink") {
                Slider(value: $ink, in: 0 ... 49.5) { Text("Ink") }
                    #if os(macOS)
                        .labelsHidden()
                    #endif
                    .accessibilityIdentifier("sketch-ink")
            }
        }
        .task {
            do {
                sourceIDs = try await loadSources()
            } catch {
                if !Task.isCancelled { sourceError = error.localizedDescription }
            }
        }
    }
}
