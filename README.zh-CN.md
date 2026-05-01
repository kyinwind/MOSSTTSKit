# MOSSTTSKit

[English](./README.md) | [简体中文](./README.zh-CN.md)

MOSSTTSKit 是一个基于 ONNX Runtime 的 MOSS-TTS-Nano Swift Package。

## 项目来源

MOSSTTSKit 是一个独立的 Swift Package，建立在开源的 MOSS-TTS-Nano 项目及相关工具链之上，目标是让 MOSS-TTS-Nano 更容易集成到 Apple 平台应用中，比如 TTSMate 和其他 Swift 项目。

上游 MOSS-TTS-Nano 项目地址：

- [OpenMOSS / MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano)

当前能力范围：

- 从 HuggingFace 自动下载并缓存 MOSS-TTS-Nano TTS 模型
- 从 HuggingFace 自动下载并缓存 MOSS Audio Tokenizer 模型
- 支持“自动下载后初始化”或“指定本地模型目录初始化”
- 加载文本 tokenizer 与音频 tokenizer
- 通过包 API 提供模型内全部内置音色，并提供 `makeSpeaker(name:referenceAudioURL:)` 语音克隆入口
- 提供基于 ONNX Runtime 的通用 `ONNXSession` 推理封装
- 提供真实模型驱动的多帧 TTS 生成路径
- 提供长文本自动切分能力，`speak(...)` 和 `speakStream(...)` 会自动分段合成、拼接，并在段间加入短暂停顿

## 致谢

这个包的实现参考和受益于以下开源项目：

- [OpenMOSS / MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) - 本包所封装的上游 TTS 模型、ONNX 推理流程和整体运行设计来源。
- [Microsoft / ONNX Runtime](https://github.com/microsoft/onnxruntime) - 用于执行导出 ONNX 模型的推理运行时。
- [microsoft / onnxruntime-swift-package-manager](https://github.com/microsoft/onnxruntime-swift-package-manager) - 本项目使用的 Swift Package 形式 ONNX Runtime 集成。
- [Hugging Face / swift-transformers](https://github.com/huggingface/swift-transformers) - 用于 tokenizer 和部分模型加载相关的 Swift 侧集成。
- [Google / SentencePiece](https://github.com/google/sentencepiece) - MOSS-TTS-Nano 使用的 tokenizer 模型格式来源。

## 许可证

本项目使用 Apache License 2.0。

- [Apache-2.0 License](./LICENSE)

## 基本用法

```swift
import MOSSTTSKit

let tts = try await MOSSTTSKit()
let result = try await tts.speak(text: "你好，欢迎使用 MOSS-TTS-Nano。")

try await tts.speakToFile(
    text: "保存成 WAV 文件。",
    outputURL: URL(fileURLWithPath: "/tmp/moss.wav")
)
```

## 长文本支持

`speak(...)`、`speakStream(...)` 和 `speakToFile(...)` 现在都可以直接处理长文本。

MOSSTTSKit 会自动：

- 把较长输入切分成多个合成 chunk
- 优先按句末标点切分，再按从句标点切分，最后按 token 预算兜底切分
- 把多个 chunk 的音频结果自动拼接起来，并在段间加入短暂停顿

切分预算由 `MOSSTTSOptions.maxTextTokensPerChunk` 控制：

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

对于较长段落，调用方通常不需要再自己手动切分文本。

重要说明：

- 对于正式的长文本合成，不建议再手动设置较小的 `maxGeneratedFrames`，除非你本来就只想做短预览。
- 当 `maxGeneratedFrames` 保持为 `nil` 时，MOSSTTSKit 会回退到模型 manifest 里的默认上限，这是普通句子和段落合成时更推荐的行为。
- 像 `8`、`16`、`32`、`64` 这样的较小帧上限，更适合 smoke test、进度 UI 联调和短句试听。

## 自动下载模型

`MOSSTTSKit()` 默认会自动下载模型。

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

默认缓存目录：

- macOS: `~/Library/Caches/MOSSTTSKit/Models`
- iOS: `<App Sandbox>/Library/Caches/MOSSTTSKit/Models`

默认下载后的目录结构：

```text
.../MOSSTTSKit/Models/
├── MOSS-TTS-Nano-100M-ONNX/
└── MOSS-Audio-Tokenizer-Nano-ONNX/
```

自定义缓存目录：

```swift
let tts = try await MOSSTTSKit(
    options: .init(
        autoDownload: true,
        cacheDir: URL(fileURLWithPath: "/path/to/custom-cache")
    )
)
```

提前预下载模型：

```swift
try await MOSSTTSKit.preload { progress in
    print(progress.description)
}
```

关闭自动下载，仅使用已缓存模型：

```swift
let tts = try await MOSSTTSKit(
    options: .init(autoDownload: false)
)
```

检查和清理缓存：

```swift
let cached = await MOSSTTSKit.isModelCached()
let cacheSize = await MOSSTTSKit.cacheSize()
try await MOSSTTSKit.clearCache()
```

## TTSMate 接入建议

推荐先用一轮最小接入测试把链路跑通。

1. 添加本地 Swift Package 依赖：

```swift
.package(path: "/Users/yangxuehui/Documents/dev/MOSSTTSKit/MOSSTTSKit")
```

2. 把 `MOSSTTSKit` 加到 TTSMate target 依赖里。

3. 第一轮接入可以二选一：

自动下载版本：

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

本地目录版本：

```swift
import MOSSTTSKit

let tts = try await MOSSTTSKit(
    ttsModelDir: URL(fileURLWithPath: "/Users/yangxuehui/Library/Caches/MOSSTTSKit/Models/MOSS-TTS-Nano-100M-ONNX"),
    audioTokenizerDir: URL(fileURLWithPath: "/Users/yangxuehui/Library/Caches/MOSSTTSKit/Models/MOSS-Audio-Tokenizer-Nano-ONNX"),
    options: MOSSTTSOptions()
)
```

4. 短句 smoke test：

```swift
let result = try await tts.speak(text: "你好，这是 TTSMate 集成测试。")
print(result.audioSamples.count)
print(result.sampleRate)
```

5. 带进度和取消：

```swift
let result = try await tts.speak(
    text: "你好，这是带进度的测试。",
    options: MOSSTTSOptions(maxGeneratedFrames: 8)
) { progress in
    print("frame \(progress.currentStep)/\(progress.totalSteps)")
    return true
}
```

6. 流式播放：

```swift
let stream = try await tts.speakStream(
    text: "这是流式播放测试。",
    options: MOSSTTSOptions(maxGeneratedFrames: 8)
)

for try await chunk in stream {
    if chunk.isFinal { break }
    // 在这里把 chunk.newAudioSamples 喂给播放器或缓冲区
    print("chunk samples:", chunk.newAudioSamples.count)
}
```

7. 获取全部内置音色：

```swift
let speakers = await tts.availableSpeakers
for speaker in speakers {
    print(speaker.identifier ?? speaker.name)
    print(speaker.displayName ?? speaker.name)
    print(speaker.group ?? "Unknown Group")
}
```

8. 用参考音频创建克隆音色：

```swift
let speaker = try await tts.makeSpeaker(
    name: "Reference",
    referenceAudioURL: URL(fileURLWithPath: "/path/to/reference.wav")
)

let cloned = try await tts.speak(
    text: "这是克隆音色测试。",
    speaker: speaker,
    options: MOSSTTSOptions(maxGeneratedFrames: 8)
)
```

接入建议：

- 第一轮 UI smoke test 或短预览，建议先用 `maxGeneratedFrames: 3` 或 `8`
- 对于正常句子和较长段落，优先使用 `MOSSTTSOptions()`，让包默认使用模型 manifest 的帧上限
- 对于较长段落，现在可以直接把完整文本交给一次 `speak(...)` 调用；只有在你想细调切分粒度时，才需要调整 `maxTextTokensPerChunk`
- 如果要边生成边播放，优先接 `speakStream(...)`
- 如果只想先确认链路能跑通，优先接 `speak(...)`
- 当前版本已经适合做集成测试，但长文本性能和更完整的播放体验还需要继续打磨

## 更多用法

限制生成帧数并观察进度：

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

完整文本合成的推荐写法：

```swift
let result = try await tts.speak(
    text: "这里可以直接放完整段落，不需要手动切分，也不需要额外设置较小的帧数上限。",
    options: MOSSTTSOptions()
)
```

流式输出音频 chunk：

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

使用本地模型目录：

```swift
let tts = try await MOSSTTSKit(
    ttsModelDir: URL(fileURLWithPath: "/path/to/MOSS-TTS-Nano-100M-ONNX"),
    audioTokenizerDir: URL(fileURLWithPath: "/path/to/MOSS-Audio-Tokenizer-Nano-ONNX")
)
```

准备克隆音色：

```swift
let speaker = try await tts.makeSpeaker(
    name: "Reference Voice",
    referenceAudioURL: URL(fileURLWithPath: "/path/to/reference.wav")
)

let cloned = try await tts.speak(text: "使用参考音频音色。", speaker: speaker)
```

## 模型文件

默认缓存目录结构：

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

内置下载器会先写入 `.partial` 临时文件，并在服务端支持 HTTP Range 时尝试断点续传。如果网络不稳定，也可以手动下载：

- `https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX`
- `https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX`

你可以：

- 让 `MOSSTTSKit()` 自动下载到默认缓存目录
- 提前调用 `MOSSTTSKit.preload(...)` 预下载
- 手动放到默认缓存目录
- 或者在初始化时传入本地模型目录

## 状态

当前包已经可以：

- 完成模型自动下载和缓存
- 支持本地模型目录初始化
- 跑真实 ONNX 多帧生成
- 支持 `speak(...)` 一次性生成
- 支持 `speakStream(...)` 流式输出
- 支持进度回调和取消

当前仍需继续完善的方向：

- 更长文本的性能优化
- 更细粒度的流式播放/缓冲策略
- 更严格的 SentencePiece tokenizer 对齐
