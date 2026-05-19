# MOSSTTSKit

[English](./README.md) | [简体中文](./README.zh-CN.md)

MOSSTTSKit is a Swift Package wrapper for MOSS-TTS-Nano ONNX models.

## Origin

MOSSTTSKit is an independent Swift Package built on top of the open-source MOSS-TTS-Nano project and related tooling. It aims to make MOSS-TTS-Nano easier to integrate into Apple-platform apps such as TTSMate and other Swift projects.

The upstream MOSS-TTS-Nano project is available here:

- [OpenMOSS / MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano)

Current package scope:

- Download and cache the MOSS-TTS-Nano TTS model files from HuggingFace.
- Download and cache the MOSS Audio Tokenizer ONNX model files from HuggingFace.
- Initialize from either cached/downloaded models or explicit local model directories.
- Load text tokenizer and audio tokenizer models.
- Expose all built-in voices from the model manifest through package APIs, plus a `makeSpeaker(name:referenceAudioURL:)` API that encodes reference audio into acoustic codes for voice cloning.
- Provide a real ONNX Runtime backed `ONNXSession` wrapper for generic tensor inference.
- Run a verified real-model preview path: prefill, global decode step, fixed frame sampler, and audio tokenizer decode for the first generated acoustic frame.
- Support automatic long-text chunking for `speak(...)` and `speakStream(...)`, with chunk concatenation and short pauses inserted between synthesized segments.

## Acknowledgements

This package builds on and benefits from the following open-source projects:

- [OpenMOSS / MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) - the upstream TTS model, ONNX runtime flow, and browser/runtime design this package is based on.
- [Microsoft / ONNX Runtime](https://github.com/microsoft/onnxruntime) - the inference runtime used to execute the exported ONNX models.
- [microsoft / onnxruntime-swift-package-manager](https://github.com/microsoft/onnxruntime-swift-package-manager) - the Swift Package integration used by this project.
- [Hugging Face / swift-transformers](https://github.com/huggingface/swift-transformers) - used for tokenizer and model-loading related Swift-side integration.
- [Google / SentencePiece](https://github.com/google/sentencepiece) - the tokenizer model format used by MOSS-TTS-Nano.

## License

This project is licensed under the Apache License 2.0.

- [Apache-2.0 License](./LICENSE)

## Usage

```swift
import MOSSTTSKit

let tts = try await MOSSTTSKit()
let result = try await tts.speak(text: "你好，欢迎使用 MOSS-TTS-Nano。")
try await tts.speakToFile(
    text: "保存成 WAV 文件。",
    outputURL: URL(fileURLWithPath: "/tmp/moss.wav")
)
```

## Long-Text Support

`speak(...)`, `speakStream(...)`, and `speakToFile(...)` now support long text directly.

MOSSTTSKit will:

- normalize risky punctuation and line breaks before tokenization
- split longer input into multiple synthesis chunks automatically
- prefer sentence-ending punctuation first, then clause punctuation, then token-budget fallback splitting
- concatenate chunk audio and insert short pauses between chunks

The chunking budget is controlled by `MOSSTTSOptions.maxTextTokensPerChunk`:

```swift
let options = MOSSTTSOptions(
    maxGeneratedFrames: 256,
    maxTextTokensPerChunk: 75
)

let result = try await tts.speak(
    text: "第一段。第二段。第三段。",
    options: options
)
```

For longer paragraphs, you usually do not need to split text manually in the caller anymore.

Important:

- Do not set a small `maxGeneratedFrames` value for long-form synthesis unless you intentionally want a short preview.
- If `maxGeneratedFrames` is left as `nil`, MOSSTTSKit will fall back to the model manifest default, which is the recommended behavior for normal sentence and paragraph synthesis.
- Small frame caps such as `8`, `16`, `32`, or `64` are best reserved for smoke tests, progress UI testing, and short preview generation.

Text preprocessing is centralized in `TextNormalizer`. It currently converts ellipses and repeated dash separators into sentence pauses, treats non-empty line breaks as boundaries, and fixes dangling final punctuation such as `Taiguanglin：`. See [docs/text-normalization.md](./docs/text-normalization.md) for the engineering rule used when adding new text rules.

## Automatic Model Download

`MOSSTTSKit()` enables automatic model download by default.

```swift
let tts = try await MOSSTTSKit(
    options: .init(
        autoDownload: true,
        progressCallback: { progress in
            print(progress.description)
        }
    )
)
```

Default cache location:

- macOS: `~/Library/Caches/MOSSTTSKit/Models`
- iOS: `<App Sandbox>/Library/Caches/MOSSTTSKit/Models`

Default downloaded folders:

```text
.../MOSSTTSKit/Models/
├── MOSS-TTS-Nano-100M-ONNX/
└── MOSS-Audio-Tokenizer-Nano-ONNX/
```

Use a custom cache directory:

```swift
let tts = try await MOSSTTSKit(
    options: .init(
        autoDownload: true,
        cacheDir: URL(fileURLWithPath: "/path/to/custom-cache")
    )
)
```

Preload models before first synthesis:

```swift
try await MOSSTTSKit.preload { progress in
    print(progress.description)
}
```

Disable automatic download and require cached models:

```swift
let tts = try await MOSSTTSKit(
    options: .init(autoDownload: false)
)
```

Check and clear cache:

```swift
let cached = await MOSSTTSKit.isModelCached()
let cacheSize = await MOSSTTSKit.cacheSize()
try await MOSSTTSKit.clearCache()
```

## TTSMate Integration

Recommended first-pass integration for TTSMate:

1. Add the Swift Package dependency:

```swift
.package(path: "/Users/yangxuehui/Documents/dev/MOSSTTSKit/MOSSTTSKit")
```

2. Add `MOSSTTSKit` to the TTSMate target dependencies.

3. For the first integration pass, either use explicit local model directories or keep auto-download on and surface the progress callback in the UI.

Auto-download version:

```swift
import MOSSTTSKit

let tts = try await MOSSTTSKit(
    options: .init(
        autoDownload: true,
        synthesisOptions: MOSSTTSOptions(),
        progressCallback: { progress in
            print(progress.description)
        }
    )
)
```

Explicit local-directory version:

```swift
import MOSSTTSKit

let tts = try await MOSSTTSKit(
    ttsModelDir: URL(fileURLWithPath: "/Users/yangxuehui/Library/Caches/MOSSTTSKit/Models/MOSS-TTS-Nano-100M-ONNX"),
    audioTokenizerDir: URL(fileURLWithPath: "/Users/yangxuehui/Library/Caches/MOSSTTSKit/Models/MOSS-Audio-Tokenizer-Nano-ONNX"),
    options: MOSSTTSOptions()
)
```

4. Smoke test with a short sentence:

```swift
let result = try await tts.speak(text: "你好，这是 TTSMate 集成测试。")
print(result.audioSamples.count)
print(result.sampleRate)
```

5. Show progress and allow cancel:

```swift
let result = try await tts.speak(
    text: "你好，这是带进度的测试。",
    options: MOSSTTSOptions(maxGeneratedFrames: 8)
) { progress in
    print("frame \(progress.currentStep)/\(progress.totalSteps)")
    return true
}
```

6. Try streaming playback:

```swift
let stream = try await tts.speakStream(
    text: "这是流式播放测试。",
    options: MOSSTTSOptions(maxGeneratedFrames: 8)
)

for try await chunk in stream {
    if chunk.isFinal { break }
    // Feed chunk.newAudioSamples into your player/buffer here.
    print("chunk samples:", chunk.newAudioSamples.count)
}
```

7. Enumerate all built-in voices:

```swift
let speakers = await tts.availableSpeakers
for speaker in speakers {
    print(speaker.identifier ?? speaker.name)
    print(speaker.displayName ?? speaker.name)
    print(speaker.group ?? "Unknown Group")
}
```

8. Build a cloned speaker from a reference WAV:

```swift
let speaker = try await tts.makeSpeaker(
    name: "Reference",
    referenceAudioURL: URL(fileURLWithPath: "/path/to/reference.wav"),
    maxDuration: 18
)

let cloned = try await tts.speak(
    text: "这是克隆音色测试。",
    speaker: speaker,
    options: MOSSTTSOptions(maxGeneratedFrames: 8)
)
```

Integration notes:

- Start with `maxGeneratedFrames: 3` or `8` only for the first UI smoke test or a deliberate short preview.
- For normal sentence and paragraph synthesis, prefer `MOSSTTSOptions()` and let the package use the model manifest frame limit by default.
- For longer paragraphs, the package now chunks text automatically. You can usually keep the full text in one `speak(...)` call and tune `maxTextTokensPerChunk` only when you need finer control.
- Voice cloning reads only the first `MOSSTTSOptions.maxReferenceAudioDuration` seconds of reference audio by default (`18` seconds), and caps the prompt at `maxReferenceAudioPromptFrames` frames by default (`220`). Pass `maxDuration:` to `makeSpeaker(...)` when you need a different reference window.
- Default auto-download cache on macOS is `~/Library/Caches/MOSSTTSKit/Models`.
- `speakStream(...)` is the best fit if TTSMate wants progressive playback.
- `speak(...)` is better for a quick blocking smoke test.
- The package currently uses a bounded real-generation path and is suitable for integration testing, but still needs longer-text performance tuning before calling it a fully polished production pipeline.

Limiting generated frames and observing progress:

```swift
let shortOptions = MOSSTTSOptions(maxGeneratedFrames: 8)
let preview = try await tts.speak(
    text: "短句测试。",
    options: shortOptions
) { progress in
    print("Generated frame \(progress.currentStep)/\(progress.totalSteps)")
    return true
}
```

Recommended default for full text synthesis:

```swift
let result = try await tts.speak(
    text: "这里可以放完整段落，不需要手动切分，也不需要额外设置低帧数上限。",
    options: MOSSTTSOptions()
)
```

Streaming audio chunks:

```swift
let stream = try await tts.speakStream(
    text: "边生成边播放。",
    options: MOSSTTSOptions(maxGeneratedFrames: 8)
)

for try await chunk in stream {
    if chunk.isFinal { break }
    print("new chunk samples:", chunk.newAudioSamples.count)
}
```

Using local model directories:

```swift
let tts = try await MOSSTTSKit(
    ttsModelDir: URL(fileURLWithPath: "/path/to/MOSS-TTS-Nano-100M-ONNX"),
    audioTokenizerDir: URL(fileURLWithPath: "/path/to/MOSS-Audio-Tokenizer-Nano-ONNX")
)
```

Preparing a cloned speaker:

```swift
let speaker = try await tts.makeSpeaker(
    name: "Reference Voice",
    referenceAudioURL: URL(fileURLWithPath: "/path/to/reference.wav"),
    maxDuration: 18
)

let cloned = try await tts.speak(text: "使用参考音频音色。", speaker: speaker)
```

## Model Files

Default cache layout:

```text
~/Library/Caches/MOSSTTSKit/Models/
├── MOSS-TTS-Nano-100M-ONNX/
│   ├── moss_tts_prefill.onnx
│   ├── moss_tts_decode_step.onnx
│   ├── moss_tts_local_decoder.onnx
│   ├── moss_tts_local_cached_step.onnx
│   ├── moss_tts_local_fixed_sampled_frame.onnx
│   ├── moss_tts_global_shared.data
│   ├── moss_tts_local_shared.data
│   ├── tokenizer.model
│   ├── tts_browser_onnx_meta.json
│   └── browser_poc_manifest.json
└── MOSS-Audio-Tokenizer-Nano-ONNX/
    ├── moss_audio_tokenizer_encode.onnx
    ├── moss_audio_tokenizer_encode.data
    ├── moss_audio_tokenizer_decode_full.onnx
    ├── moss_audio_tokenizer_decode_step.onnx
    ├── moss_audio_tokenizer_decode_shared.data
    └── codec_browser_onnx_meta.json
```

The built-in downloader now writes `.partial` files and resumes interrupted downloads when the server supports HTTP range requests. If the network is unstable, you can manually download the files from:

- `https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX`
- `https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX`

Then either place them in the default cache layout above, let `MOSSTTSKit()` auto-download into that cache, or pass explicit directories to:

```swift
let tts = try await MOSSTTSKit(
    ttsModelDir: URL(fileURLWithPath: "/path/to/MOSS-TTS-Nano-100M-ONNX"),
    audioTokenizerDir: URL(fileURLWithPath: "/path/to/MOSS-Audio-Tokenizer-Nano-ONNX")
)
```

For now, model inspection is an internal development helper and is not exported as a package product. App integrations such as TTSMate only need the library product `MOSSTTSKit`.

## Status

The package builds and its unit tests pass. ONNX Runtime is integrated for generic sessions, the audio tokenizer path calls real ORT sessions, and real-model integration tests cover the first generated acoustic frame.

The package also parses `browser_poc_manifest.json`, builds the MOSS-TTS prefill request rows that combine text tokens and reference audio codes, and exposes `MOSSTTSEngine.generateFirstAudioFrame(...)` as a small real-generation slice. `MOSSTTSEngine.generateAudioCodes(...)` runs a bounded multi-frame continuation loop. `MOSSTTSKit.speak(...)` now uses that real multi-frame path. The default `MOSSTTSOptions.maxGeneratedFrames` is 32 for early integration safety; callers can raise or lower it. `MOSSTTSKit.speakStream(...)` exposes decoded audio chunks as an `AsyncThrowingStream`.

The ONNX model repository ships `tokenizer.model` without `tokenizer.json` / `tokenizer_config.json`; MOSSTTSKit currently falls back to the model's byte-token range for offline local initialization. A true SentencePiece parser is still needed for exact tokenizer parity.

The remaining major task is to make the full MOSS-TTS autoregressive inference loop production-ready by adding streaming audio output, validating longer generation performance, and polishing:

- `moss_tts_decode_step.onnx`
- `moss_tts_local_decoder.onnx`
- `moss_tts_local_cached_step.onnx`
- `moss_tts_local_fixed_sampled_frame.onnx`
