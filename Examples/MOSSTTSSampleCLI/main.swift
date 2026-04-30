import Foundation
import MOSSTTSKit

@main
struct MOSSTTSSampleCLI {
    static func main() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        let text = arguments.first ?? "你好，欢迎使用 MOSS TTS。"
        let outputPath = arguments.dropFirst().first ?? "/tmp/mosstts-sample.wav"

        let downloader = ModelDownloader()
        let ttsDir = await downloader.ttsModelDir(for: .mossTTSNano100M)
        let audioTokenizerDir = await downloader.tokenizerModelDir(for: .mossTTSNano100M)

        let tts = try await MOSSTTSKit(
            ttsModelDir: ttsDir,
            audioTokenizerDir: audioTokenizerDir,
            options: MOSSTTSOptions(maxGeneratedFrames: 64)
        )

        let speakers = await tts.availableSpeakers
        let speaker = speakers.first

        let outputURL = URL(fileURLWithPath: outputPath)
        let result = try await tts.speak(text: text, speaker: speaker)
        try await tts.speakToFile(text: text, outputURL: outputURL, speaker: speaker)

        print("wrote \(outputURL.path)")
        print("sampleRate=\(result.sampleRate)")
        print("channels=\(result.channels)")
        print("duration=\(result.duration)")
        print("samples=\(result.audioSamples.count)")
        print("speaker=\(speaker?.displayName ?? speaker?.name ?? "none")")
    }
}
