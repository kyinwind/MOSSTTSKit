import Foundation
import MOSSTTSKit

let arguments = Array(CommandLine.arguments.dropFirst())

func printUsage() {
    print("""
    Usage:
      mosstts-inspect
      mosstts-inspect <tts-model-dir> <audio-tokenizer-dir>
    
    Without arguments, the default MOSSTTSKit cache is inspected:
      ~/Library/Caches/MOSSTTSKit/Models/MOSS-TTS-Nano-100M-ONNX
      ~/Library/Caches/MOSSTTSKit/Models/MOSS-Audio-Tokenizer-Nano-ONNX
    """)
}

let modelPaths: ModelPaths

switch arguments.count {
case 0:
    let downloader = ModelDownloader()
    let ttsModelDir = await downloader.ttsModelDir(for: .mossTTSNano100M)
    let audioTokenizerDir = await downloader.tokenizerModelDir(for: .mossTTSNano100M)
    modelPaths = ModelPaths(ttsModelDir: ttsModelDir, audioTokenizerDir: audioTokenizerDir)
case 2:
    modelPaths = ModelPaths(
        ttsModelDir: URL(fileURLWithPath: arguments[0]),
        audioTokenizerDir: URL(fileURLWithPath: arguments[1])
    )
default:
    printUsage()
    exit(64)
}

let availability = modelPaths.availability()
if !availability.isComplete {
    print("Model files are incomplete:\n")
    print(availability.missingFilesDescription)
    exit(66)
}

do {
    let report = try ONNXModelInspector.markdownReport(for: modelPaths)
    print(report)
} catch {
    print("Failed to inspect ONNX models: \(error.localizedDescription)")
    exit(1)
}
