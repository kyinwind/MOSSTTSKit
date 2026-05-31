import Foundation
import MOSSTTSKit

@main
struct MOSSCloneExperimentCLI {
    private struct Arguments {
        var textFile: String?
        var referenceAudio: String?
        var outputDirectory = "/tmp/mosstts-clone-experiment"
    }

    private struct Experiment {
        let id: String
        let description: String
        let speaker: MOSSSpeaker?
        let options: MOSSTTSOptions
    }

    static func main() async throws {
        let arguments = try parseArguments()
        guard let textFile = arguments.textFile else {
            throw CLIError.missingArgument("--text-file")
        }
        guard let referenceAudio = arguments.referenceAudio else {
            throw CLIError.missingArgument("--ref")
        }

        let textURL = URL(fileURLWithPath: textFile)
        let referenceURL = URL(fileURLWithPath: referenceAudio)
        let outputURL = URL(fileURLWithPath: arguments.outputDirectory, isDirectory: true)

        let text = try String(contentsOf: textURL, encoding: .utf8)
        try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)

        let downloader = ModelDownloader()
        let ttsDir = await downloader.ttsModelDir(for: .mossTTSNano100M)
        let audioTokenizerDir = await downloader.tokenizerModelDir(for: .mossTTSNano100M)

        let baseOptions = MOSSTTSOptions(
            temperature: 0.6,
            topK: 50,
            maxLength: 10000,
            maxGeneratedFrames: nil,
            maxTextTokensPerChunk: 75,
            seed: 1234,
            batchSize: 1,
            maxReferenceAudioDuration: 18,
            maxReferenceAudioPromptFrames: 220,
            sampleRate: 48000,
            channels: 2,
            enableSmoothing: true,
            precision: .float32,
            useCoreML: true,
            numThreads: 4
        )

        let tts = try await MOSSTTSKit(
            ttsModelDir: ttsDir,
            audioTokenizerDir: audioTokenizerDir,
            options: baseOptions
        )

        let builtInSpeaker = await tts.availableSpeakers.first
        let cloneAuto = try await tts.makeSpeaker(name: "clone_auto_reference", referenceAudioURL: referenceURL)
        let cloneFullLegacy = try await tts.makeSpeaker(
            name: "clone_full_reference_legacy",
            referenceAudioURL: referenceURL,
            referenceAudioProcessing: .none
        )
        let cloneMax3 = try await tts.makeSpeaker(
            name: "clone_max_3s_reference",
            referenceAudioURL: referenceURL,
            maxDuration: 3,
            referenceAudioProcessing: .none
        )
        let cloneMax2 = try await tts.makeSpeaker(
            name: "clone_max_2s_reference",
            referenceAudioURL: referenceURL,
            maxDuration: 2,
            referenceAudioProcessing: .none
        )

        var frames80Options = baseOptions
        frames80Options.maxReferenceAudioPromptFrames = 80
        var frames120Options = baseOptions
        frames120Options.maxReferenceAudioPromptFrames = 120

        let experiments: [Experiment] = [
            Experiment(
                id: "00_builtin_baseline",
                description: "First built-in speaker baseline",
                speaker: builtInSpeaker,
                options: baseOptions
            ),
            Experiment(
                id: "01_clone_auto",
                description: "Clone with automatic reference segment selection",
                speaker: cloneAuto,
                options: baseOptions
            ),
            Experiment(
                id: "02_clone_legacy_full",
                description: "Clone with the full allowed reference prompt and no preprocessing",
                speaker: cloneFullLegacy,
                options: baseOptions
            ),
            Experiment(
                id: "03_clone_legacy_max3s",
                description: "Clone with reference audio capped at 3 seconds",
                speaker: cloneMax3,
                options: baseOptions
            ),
            Experiment(
                id: "04_clone_legacy_max2s",
                description: "Clone with reference audio capped at 2 seconds",
                speaker: cloneMax2,
                options: baseOptions
            ),
            Experiment(
                id: "05_clone_auto_frames80",
                description: "Clone with prompt capped to 80 acoustic-code frames",
                speaker: cloneAuto,
                options: frames80Options
            ),
            Experiment(
                id: "06_clone_auto_frames120",
                description: "Clone with prompt capped to 120 acoustic-code frames",
                speaker: cloneAuto,
                options: frames120Options
            )
        ]

        var index: [[String: Any]] = []

        for experiment in experiments {
            let fileURL = outputURL.appendingPathComponent("\(experiment.id).wav")
            print("running \(experiment.id): \(experiment.description)")

            let result = try await tts.speak(
                text: text,
                speaker: experiment.speaker,
                options: experiment.options
            )
            try MOSSAudioExporter.exportWAV(
                samples: result.audioSamples,
                sampleRate: result.sampleRate,
                channels: result.channels,
                to: fileURL.path
            )

            index.append([
                "id": experiment.id,
                "description": experiment.description,
                "filename": fileURL.lastPathComponent,
                "duration": result.duration,
                "sample_rate": result.sampleRate,
                "channels": result.channels,
                "speaker": experiment.speaker?.displayName ?? experiment.speaker?.name ?? "none",
                "prompt_frames": experiment.options.maxReferenceAudioPromptFrames as Any
            ])

            print("wrote \(fileURL.path)")
        }

        let indexURL = outputURL.appendingPathComponent("index.json")
        let indexData = try JSONSerialization.data(withJSONObject: index, options: [.prettyPrinted, .sortedKeys])
        try indexData.write(to: indexURL)
        print("wrote \(indexURL.path)")
    }

    private static func parseArguments() throws -> Arguments {
        var result = Arguments()
        var iterator = Array(CommandLine.arguments.dropFirst()).makeIterator()

        while let argument = iterator.next() {
            switch argument {
            case "--text-file":
                result.textFile = iterator.next()
            case "--ref":
                result.referenceAudio = iterator.next()
            case "--output-dir":
                result.outputDirectory = iterator.next() ?? result.outputDirectory
            case "--help", "-h":
                printUsage()
                Foundation.exit(0)
            default:
                throw CLIError.unknownArgument(argument)
            }
        }

        return result
    }

    private static func printUsage() {
        print("""
        Usage:
          swift run MOSSCloneExperimentCLI --text-file <path> --ref <wav> [--output-dir <dir>]
        """)
    }
}

private enum CLIError: Error, LocalizedError {
    case missingArgument(String)
    case unknownArgument(String)

    var errorDescription: String? {
        switch self {
        case .missingArgument(let name):
            return "Missing required argument: \(name)"
        case .unknownArgument(let name):
            return "Unknown argument: \(name)"
        }
    }
}
