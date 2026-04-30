import Foundation
import MOSSTTSKit

@main
struct MOSSFrameDumpCLI {
    static func main() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        let text = arguments.first ?? "你好，这是一个包内测试音频。"

        let downloader = ModelDownloader()
        let ttsDir = await downloader.ttsModelDir(for: .mossTTSNano100M)
        let audioTokenizerDir = await downloader.tokenizerModelDir(for: .mossTTSNano100M)

        let tts = try await MOSSTTSKit(
            ttsModelDir: ttsDir,
            audioTokenizerDir: audioTokenizerDir,
            options: MOSSTTSOptions(maxGeneratedFrames: 64, seed: 1234)
        )

        guard let manifest = await tts.browserManifest else {
            throw NSError(domain: "MOSSFrameDumpCLI", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "browser_poc_manifest.json not found"
            ])
        }

        let encoding = try await tts.textTokenizer.encode(text)
        let textTokenIds = encoding.ids.map { Int32($0) }

        let speakers = await tts.availableSpeakers
        guard let speaker = speakers.first, let promptAudioCodes = speaker.referenceAudioCodes else {
            throw NSError(domain: "MOSSFrameDumpCLI", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "No builtin speaker prompt audio codes available"
            ])
        }

        let result = try await tts.engine.generateAudioCodes(
            textTokenIds: textTokenIds,
            promptAudioCodes: promptAudioCodes,
            manifest: manifest,
            maxFrames: 64,
            seed: 1234
        )

        let payload: [String: Any] = [
            "speaker": speaker.identifier ?? speaker.name,
            "text": text,
            "text_token_ids": textTokenIds.map(Int.init),
            "frame_count": result.audioCodes.count,
            "did_reach_stop": result.didReachStop,
            "frames": result.audioCodes.map { $0.map(Int.init) }
        ]

        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        print(String(decoding: data, as: UTF8.self))
    }
}
