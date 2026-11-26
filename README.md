# PenGPTSwift

An independent Swift port of [PenGPT](https://github.com/greydanus/pengpt) for iOS and macOS. It uses Accelerate to turn text into ordered handwriting or sketch strokes on-device. Render the strokes with SwiftUI or Core Graphics, or export them as SVG.

## Demo

Handwriting and text-to-sketch on iPhone Duo.

https://github.com/user-attachments/assets/8af83e17-8e30-45f4-a8bc-5fb1b9065ce4

## Capabilities

| Capability | API |
| --- | --- |
| Handwriting with a preset writer | `HandwritingModel.write` |
| Incremental handwriting with layout | `HandwritingModel.writeWords` |
| Text-to-sketch with the upstream TinyCLIP encoder | `SketchModel.draw` |
| Progressive sketches with live guidance, ink, and stroke-count controls | `SketchModel.startSketch` |
| SVG export | `InkDrawing.svg` |
| Write with an imported style fitted by upstream tools | `HandwritingStyle` |

## Install

Use iOS 16 or later, macOS 13 or later, and a Swift 6 toolchain. In Xcode, add `https://github.com/qiyangdev/PenGPTSwift.git` under **Add Package Dependencies**, then link the **PenGPT** product.

For a Swift package, add the dependency and link its product in your target:

```swift
.package(url: "https://github.com/qiyangdev/PenGPTSwift.git", from: "0.1.0")

.product(name: "PenGPT", package: "PenGPTSwift")
```

## Prepare the model files

Run these commands on macOS from a checkout of this repository. The converter uses the [upstream models-2026-11-16 release](https://github.com/greydanus/pengpt/releases/tag/models-2026-11-16).

```sh
curl -fLO https://github.com/greydanus/pengpt/releases/download/models-2026-11-16/hand.npz
swift run -c release pengpt-convert hand.npz hand.pengpt
```

For sketches, also convert the decoder and text encoder:

```sh
curl -fLO https://github.com/greydanus/pengpt/releases/download/models-2026-11-16/pengpt.npz
swift run -c release pengpt-convert --component sketch pengpt.npz sketch.pengpt
swift run -c release pengpt-convert --component words pengpt.npz words.pengpt
```

Pass local file URLs to the Swift runtime. You can place the converted files in your app bundle or load them from a local model cache. The models are downloaded separately from the source package.

## Use the package

```swift
import PenGPT

let model = try await HandwritingModel.load(weightsAt: handURL)
let ink = try await model.write("The quick brown fox", writer: "sam")
let svg = try ink.svg()
```

`model.writerIDs` lists the available presets. For handwriting animation, iterate `writeWords` and append each word's strokes. All words share drawing coordinates, with positive y pointing down and the first baseline at y = 0.

```swift
for try await word in try model.writeWords("The quick brown fox", writer: "sam") {
    // Append or animate word.strokes.
}
```

Sketch sessions return cumulative snapshots. Replace the displayed drawing with each snapshot. Pass `SketchControls` to `nextFrame(controls:)` to adjust the rest of a sketch during generation.

```swift
let sketch = try await SketchModel.load(weightsAt: sketchURL, textEncoderAt: wordsURL)
let session = try await sketch.startSketch("a cat on a windowsill")
while let frame = try await session.nextFrame() {
    // Display frame.drawing.
}
```

If you already fitted a handwriting style with the [upstream photo workflow](https://github.com/greydanus/pengpt#setup-and-training), convert its `.npy` output with `swift run pengpt-convert --style me.npy me.json` and load it with `HandwritingStyle.load(from:)`. Photo tracing and fitting run through the upstream tool.

## Run the iOS example

Copy `hand.pengpt` into `Examples/Handwriting/Models/`. Add `sketch.pengpt` and `words.pengpt` to enable Draw mode. Open [HandwritingExample.xcodeproj](Examples/Handwriting/HandwritingExample.xcodeproj) and run the **HandwritingExample** scheme on an iOS simulator.

Use **Choose Writer** in the Write toolbar to search the model’s writer IDs and regenerate with the selected style. In Draw mode, open **Drawing Settings** in the toolbar to choose the source for the next drawing and adjust guidance and ink. The example also supports pause and resume and SVG sharing. Both modes share a full-width transparent canvas and a text field in the bottom toolbar.

## Run the macOS example

Use the same model files as the iOS example. In Xcode, select the **HandwritingExampleMac** scheme and run it on **My Mac**. The macOS example requires macOS 14 or later.

Use the inspector to switch between Write and Draw, enter text, and control playback. In Write mode, the Writer menu shows the selected style and lets you choose a different one. In Draw mode, the inspector shows the source, guidance, and ink settings. Use the window toolbar to share the result or toggle the inspector.

## Lint and format

Install SwiftLint with `brew install swiftlint`. Formatting uses `swift-format` from the active Xcode toolchain. The formatting baseline and CI use Xcode 27.

```sh
make lint          # SwiftLint checks
make format        # Rewrite Swift files using .swift-format
make format-check  # Check formatting without rewriting files
make check         # Run both checks, as CI does
```

These commands cover the Package manifest, sources, tests, and the handwriting example.

## Tests

Run `swift test -c release` on macOS. To run the comparisons against released models, put the converted files in `Tests/PenGPTTests/Fixtures/` as `released-hand.pengpt`, `released-sketch.pengpt`, and `released-words.pengpt`. CI also runs these comparisons.

## Credits

PenGPTSwift is an independent port of [PenGPT by Sam Greydanus and contributors](https://github.com/greydanus/pengpt), based on [revision 3637f5c](https://github.com/greydanus/pengpt/commit/3637f5c29989cd511bc3af90e3af1d2cbcdcb3e5). See [LICENSE](LICENSE) and [NOTICE](NOTICE) for source and model attribution.
