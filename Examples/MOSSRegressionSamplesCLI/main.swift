import Foundation
import MOSSTTSKit

@main
struct MOSSRegressionSamplesCLI {
    private struct Sample: Sendable {
        let identifier: String
        let text: String
    }

    private static let samples: [Sample] = [
        .init(identifier: "01_basic_cn", text: "你好，这是一个包内测试音频。"),
        .init(identifier: "02_date_weather", text: "今天是2026年5月1日，天气不错。"),
        .init(identifier: "03_timer_reminder", text: "请在3分钟后提醒我开会。"),
        .init(identifier: "04_mixed_cn_en", text: "OpenAI 的 GPT 模型现在支持中英混排。"),
        .init(identifier: "05_price_decimal", text: "价格是 12.5 元，不是 15 元。"),
        .init(identifier: "06_multisentence_mix", text: "第一段说中文。Second sentence mixes English.")
    ]

    static func main() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        let outputPath = arguments.first ?? "/tmp/mosstts-regression-samples"
        let outputURL = URL(fileURLWithPath: outputPath, isDirectory: true)

        try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)

        let downloader = ModelDownloader()
        let ttsDir = await downloader.ttsModelDir(for: .mossTTSNano100M)
        let audioTokenizerDir = await downloader.tokenizerModelDir(for: .mossTTSNano100M)

        let tts = try await MOSSTTSKit(
            ttsModelDir: ttsDir,
            audioTokenizerDir: audioTokenizerDir,
            options: MOSSTTSOptions()
        )

        let speakers = await tts.availableSpeakers
        let speaker = speakers.first

        var index: [[String: Any]] = []
        for sample in samples {
            let fileURL = outputURL.appendingPathComponent("\(sample.identifier).wav")
            let result = try await tts.speak(text: sample.text, speaker: speaker)
            try await tts.speakToFile(text: sample.text, outputURL: fileURL, speaker: speaker)

            index.append([
                "id": sample.identifier,
                "text": sample.text,
                "filename": fileURL.lastPathComponent,
                "duration": result.duration,
                "sample_rate": result.sampleRate,
                "channels": result.channels,
                "speaker": speaker?.displayName ?? speaker?.name ?? "none"
            ])

            print("wrote \(fileURL.path)")
        }

        let indexURL = outputURL.appendingPathComponent("index.json")
        let indexData = try JSONSerialization.data(withJSONObject: index, options: [.prettyPrinted, .sortedKeys])
        try indexData.write(to: indexURL)
        print("wrote \(indexURL.path)")
    }
}
