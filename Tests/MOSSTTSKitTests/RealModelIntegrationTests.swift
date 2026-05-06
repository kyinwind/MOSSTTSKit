import XCTest
@testable import MOSSTTSKit

final class RealModelIntegrationTests: XCTestCase {
    func testEllipsisIsNormalizedBeforeSynthesisWhenModelsAreAvailable() async throws {
        let downloader = ModelDownloader()
        let ttsDir = await downloader.ttsModelDir(for: .mossTTSNano100M)
        let audioTokenizerDir = await downloader.tokenizerModelDir(for: .mossTTSNano100M)
        let paths = ModelPaths(ttsModelDir: ttsDir, audioTokenizerDir: audioTokenizerDir)

        guard paths.availability().isComplete else {
            throw XCTSkip("MOSS-TTS model files are not available in the default cache")
        }

        _ = try await MOSSTTSKit(
            ttsModelDir: ttsDir,
            audioTokenizerDir: audioTokenizerDir,
            options: MOSSTTSOptions(maxGeneratedFrames: 1)
        )
        let processed = TextNormalizer().normalize("""
        利娜正睡在我身边，她的双手握着，就像她平常睡觉时那样……

        我一点都不想再睡了。
        """)

        XCTAssertFalse(processed.contains("…"))
        XCTAssertTrue(processed.contains("那样。 我一点"))
        XCTAssertFalse(processed.contains("。。"))

        let speakerLabel = TextNormalizer().normalize("Taiguanglin：")
        XCTAssertEqual(speakerLabel, "Taiguanglin.")
    }

    func testDecodeBuiltinVoiceCodesWhenModelsAreAvailable() async throws {
        let downloader = ModelDownloader()
        let ttsDir = await downloader.ttsModelDir(for: .mossTTSNano100M)
        let audioTokenizerDir = await downloader.tokenizerModelDir(for: .mossTTSNano100M)
        let paths = ModelPaths(ttsModelDir: ttsDir, audioTokenizerDir: audioTokenizerDir)
        
        guard paths.availability().isComplete else {
            throw XCTSkip("MOSS-TTS model files are not available in the default cache")
        }
        
        guard let manifest = try MOSSBrowserManifest.find(in: ttsDir),
              let promptCodes = manifest.builtinVoices.first?.promptAudioCodes,
              !promptCodes.isEmpty else {
            throw XCTSkip("No builtin voice prompt audio codes found")
        }
        
        let tokenizer = try await AudioTokenizerONNX.fromDirectory(audioTokenizerDir.path)
        let samples = try await tokenizer.decode(codes: Array(promptCodes.prefix(8)))
        
        XCTAssertFalse(samples.isEmpty)
        XCTAssertEqual(samples.count % tokenizer.numChannels, 0)
    }
    
    func testPrefillWithBuiltinVoiceWhenModelsAreAvailable() async throws {
        let downloader = ModelDownloader()
        let ttsDir = await downloader.ttsModelDir(for: .mossTTSNano100M)
        let audioTokenizerDir = await downloader.tokenizerModelDir(for: .mossTTSNano100M)
        let paths = ModelPaths(ttsModelDir: ttsDir, audioTokenizerDir: audioTokenizerDir)
        
        guard paths.availability().isComplete else {
            throw XCTSkip("MOSS-TTS model files are not available in the default cache")
        }
        
        guard let manifest = try MOSSBrowserManifest.find(in: ttsDir),
              let promptCodes = manifest.builtinVoices.first?.promptAudioCodes,
              !promptCodes.isEmpty else {
            throw XCTSkip("No builtin voice prompt audio codes found")
        }
        
        let builder = MOSSInferenceRequestBuilder(manifest: manifest)
        let rows = builder.buildVoiceCloneRequestRows(
            promptAudioCodes: Array(promptCodes.prefix(4)),
            textTokenIds: [8651, 2691]
        )
        
        let engine = try await MOSSTTSEngine(modelDir: ttsDir)
        let prefill = try await engine.runPrefill(
            inputIds: rows.inputIds,
            attentionMask: rows.attentionMask
        )
        
        XCTAssertEqual(prefill.globalHiddenShape, [1, rows.inputIds.count, 768])
        XCTAssertEqual(prefill.sequenceLength, rows.inputIds.count)
        XCTAssertEqual(prefill.keyValues.count, 24)
        XCTAssertNotNil(prefill.keyValues["present_key_0"])
        XCTAssertNotNil(prefill.keyValues["present_value_11"])
    }
    
    func testDecodeStepAfterPrefillWhenModelsAreAvailable() async throws {
        let downloader = ModelDownloader()
        let ttsDir = await downloader.ttsModelDir(for: .mossTTSNano100M)
        let audioTokenizerDir = await downloader.tokenizerModelDir(for: .mossTTSNano100M)
        let paths = ModelPaths(ttsModelDir: ttsDir, audioTokenizerDir: audioTokenizerDir)
        
        guard paths.availability().isComplete else {
            throw XCTSkip("MOSS-TTS model files are not available in the default cache")
        }
        
        guard let manifest = try MOSSBrowserManifest.find(in: ttsDir),
              let promptCodes = manifest.builtinVoices.first?.promptAudioCodes,
              !promptCodes.isEmpty else {
            throw XCTSkip("No builtin voice prompt audio codes found")
        }
        
        let builder = MOSSInferenceRequestBuilder(manifest: manifest)
        let rows = builder.buildVoiceCloneRequestRows(
            promptAudioCodes: Array(promptCodes.prefix(4)),
            textTokenIds: [8651, 2691]
        )
        let nextRows = builder.buildAudioPrefixRows(
            promptAudioCodes: [[Int32](repeating: Int32(manifest.ttsConfig.audioPadTokenId), count: manifest.ttsConfig.nVq)],
            slotTokenId: manifest.ttsConfig.audioAssistantSlotTokenId
        )
        
        let engine = try await MOSSTTSEngine(modelDir: ttsDir)
        let prefill = try await engine.runPrefill(
            inputIds: rows.inputIds,
            attentionMask: rows.attentionMask
        )
        let decode = try await engine.runDecodeStep(
            inputIds: nextRows,
            pastValidLength: Int32(prefill.sequenceLength),
            previousKeyValues: prefill.keyValues
        )
        
        XCTAssertEqual(decode.globalHiddenShape, [1, 1, 768])
        XCTAssertEqual(decode.totalSequenceLength, prefill.sequenceLength + 1)
        XCTAssertEqual(decode.keyValues.count, 24)
        XCTAssertNotNil(decode.keyValues["present_key_0"])
        XCTAssertNotNil(decode.keyValues["present_value_11"])
    }
    
    func testFixedSamplerAfterDecodeStepWhenModelsAreAvailable() async throws {
        let downloader = ModelDownloader()
        let ttsDir = await downloader.ttsModelDir(for: .mossTTSNano100M)
        let audioTokenizerDir = await downloader.tokenizerModelDir(for: .mossTTSNano100M)
        let paths = ModelPaths(ttsModelDir: ttsDir, audioTokenizerDir: audioTokenizerDir)
        
        guard paths.availability().isComplete else {
            throw XCTSkip("MOSS-TTS model files are not available in the default cache")
        }
        
        guard let manifest = try MOSSBrowserManifest.find(in: ttsDir),
              let promptCodes = manifest.builtinVoices.first?.promptAudioCodes,
              !promptCodes.isEmpty else {
            throw XCTSkip("No builtin voice prompt audio codes found")
        }
        
        let builder = MOSSInferenceRequestBuilder(manifest: manifest)
        let rows = builder.buildVoiceCloneRequestRows(
            promptAudioCodes: Array(promptCodes.prefix(4)),
            textTokenIds: [8651, 2691]
        )
        let nextRows = builder.buildAudioPrefixRows(
            promptAudioCodes: [[Int32](repeating: Int32(manifest.ttsConfig.audioPadTokenId), count: manifest.ttsConfig.nVq)],
            slotTokenId: manifest.ttsConfig.audioAssistantSlotTokenId
        )
        
        let engine = try await MOSSTTSEngine(modelDir: ttsDir)
        let prefill = try await engine.runPrefill(
            inputIds: rows.inputIds,
            attentionMask: rows.attentionMask
        )
        let decode = try await engine.runDecodeStep(
            inputIds: nextRows,
            pastValidLength: Int32(prefill.sequenceLength),
            previousKeyValues: prefill.keyValues
        )
        
        let sampled = try await engine.runFixedSampledFrame(
            globalHidden: decode.globalHidden,
            assistantRandomU: 0.5,
            audioRandomU: [Float](repeating: 0.5, count: 16)
        )
        
        XCTAssertEqual(sampled.frameTokenIds.count, 16)
        XCTAssertTrue(sampled.frameTokenIds.allSatisfy { $0 >= 0 && $0 < 1024 })
    }
    
    func testOneSampledFrameDecodesToAudioWhenModelsAreAvailable() async throws {
        let downloader = ModelDownloader()
        let ttsDir = await downloader.ttsModelDir(for: .mossTTSNano100M)
        let audioTokenizerDir = await downloader.tokenizerModelDir(for: .mossTTSNano100M)
        let paths = ModelPaths(ttsModelDir: ttsDir, audioTokenizerDir: audioTokenizerDir)
        
        guard paths.availability().isComplete else {
            throw XCTSkip("MOSS-TTS model files are not available in the default cache")
        }
        
        guard let manifest = try MOSSBrowserManifest.find(in: ttsDir),
              let promptCodes = manifest.builtinVoices.first?.promptAudioCodes,
              !promptCodes.isEmpty else {
            throw XCTSkip("No builtin voice prompt audio codes found")
        }
        
        let builder = MOSSInferenceRequestBuilder(manifest: manifest)
        let rows = builder.buildVoiceCloneRequestRows(
            promptAudioCodes: Array(promptCodes.prefix(4)),
            textTokenIds: [8651, 2691]
        )
        let nextRows = builder.buildAudioPrefixRows(
            promptAudioCodes: [[Int32](repeating: Int32(manifest.ttsConfig.audioPadTokenId), count: manifest.ttsConfig.nVq)],
            slotTokenId: manifest.ttsConfig.audioAssistantSlotTokenId
        )
        
        let engine = try await MOSSTTSEngine(modelDir: ttsDir)
        let prefill = try await engine.runPrefill(
            inputIds: rows.inputIds,
            attentionMask: rows.attentionMask
        )
        let decode = try await engine.runDecodeStep(
            inputIds: nextRows,
            pastValidLength: Int32(prefill.sequenceLength),
            previousKeyValues: prefill.keyValues
        )
        let sampled = try await engine.runFixedSampledFrame(
            globalHidden: decode.globalHidden,
            assistantRandomU: 0.5,
            audioRandomU: [Float](repeating: 0.5, count: 16)
        )
        
        let tokenizer = try await AudioTokenizerONNX.fromDirectory(audioTokenizerDir.path)
        let samples = try await tokenizer.decode(codes: [sampled.frameTokenIds])
        
        XCTAssertFalse(samples.isEmpty)
        XCTAssertEqual(samples.count % tokenizer.numChannels, 0)
    }
    
    func testGenerateFirstAudioFrameWhenModelsAreAvailable() async throws {
        let downloader = ModelDownloader()
        let ttsDir = await downloader.ttsModelDir(for: .mossTTSNano100M)
        let audioTokenizerDir = await downloader.tokenizerModelDir(for: .mossTTSNano100M)
        let paths = ModelPaths(ttsModelDir: ttsDir, audioTokenizerDir: audioTokenizerDir)
        
        guard paths.availability().isComplete else {
            throw XCTSkip("MOSS-TTS model files are not available in the default cache")
        }
        
        guard let manifest = try MOSSBrowserManifest.find(in: ttsDir),
              let promptCodes = manifest.builtinVoices.first?.promptAudioCodes,
              !promptCodes.isEmpty else {
            throw XCTSkip("No builtin voice prompt audio codes found")
        }
        
        let engine = try await MOSSTTSEngine(modelDir: ttsDir)
        let frame = try await engine.generateFirstAudioFrame(
            textTokenIds: [8651, 2691],
            promptAudioCodes: Array(promptCodes.prefix(4)),
            manifest: manifest,
            assistantRandomU: 0.5,
            audioRandomU: [Float](repeating: 0.5, count: manifest.ttsConfig.nVq)
        )
        
        XCTAssertEqual(frame.audioCodes.count, 1)
        XCTAssertEqual(frame.audioCodes[0].count, 16)
        XCTAssertTrue(frame.audioCodes[0].allSatisfy { $0 >= 0 && $0 < 1024 })
        XCTAssertGreaterThan(frame.prefillSequenceLength, 0)
        XCTAssertEqual(frame.totalSequenceLength, frame.prefillSequenceLength)
    }
    
    func testGenerateMultipleAudioCodeFramesWhenModelsAreAvailable() async throws {
        let downloader = ModelDownloader()
        let ttsDir = await downloader.ttsModelDir(for: .mossTTSNano100M)
        let audioTokenizerDir = await downloader.tokenizerModelDir(for: .mossTTSNano100M)
        let paths = ModelPaths(ttsModelDir: ttsDir, audioTokenizerDir: audioTokenizerDir)
        
        guard paths.availability().isComplete else {
            throw XCTSkip("MOSS-TTS model files are not available in the default cache")
        }
        
        guard let manifest = try MOSSBrowserManifest.find(in: ttsDir),
              let promptCodes = manifest.builtinVoices.first?.promptAudioCodes,
              !promptCodes.isEmpty else {
            throw XCTSkip("No builtin voice prompt audio codes found")
        }
        
        let engine = try await MOSSTTSEngine(modelDir: ttsDir)
        let result = try await engine.generateAudioCodes(
            textTokenIds: [8651, 2691],
            promptAudioCodes: Array(promptCodes.prefix(4)),
            manifest: manifest,
            maxFrames: 3,
            assistantRandomU: 0.5,
            audioRandomU: [Float](repeating: 0.5, count: manifest.ttsConfig.nVq)
        )
        
        XCTAssertEqual(result.audioCodes.count, 3)
        XCTAssertEqual(result.audioCodes.flatMap { $0 }.count, 48)
        XCTAssertTrue(result.audioCodes.flatMap { $0 }.allSatisfy { $0 >= 0 && $0 < 1024 })
        XCTAssertEqual(result.totalSequenceLength, result.prefillSequenceLength + result.audioCodes.count)
        
        let tokenizer = try await AudioTokenizerONNX.fromDirectory(audioTokenizerDir.path)
        let samples = try await tokenizer.decode(codes: result.audioCodes)
        XCTAssertFalse(samples.isEmpty)
        XCTAssertEqual(samples.count % tokenizer.numChannels, 0)
    }
    
    func testSpeakUsesRealPreviewPathWhenModelsAreAvailable() async throws {
        let downloader = ModelDownloader()
        let ttsDir = await downloader.ttsModelDir(for: .mossTTSNano100M)
        let audioTokenizerDir = await downloader.tokenizerModelDir(for: .mossTTSNano100M)
        let paths = ModelPaths(ttsModelDir: ttsDir, audioTokenizerDir: audioTokenizerDir)
        
        guard paths.availability().isComplete else {
            throw XCTSkip("MOSS-TTS model files are not available in the default cache")
        }
        
        let tts = try await MOSSTTSKit(
            ttsModelDir: ttsDir,
            audioTokenizerDir: audioTokenizerDir,
            options: MOSSTTSOptions(maxLength: 3)
        )
        let result = try await tts.speak(text: "你好")
        
        XCTAssertFalse(result.audioSamples.isEmpty)
        XCTAssertEqual(result.sampleRate, 48_000)
        XCTAssertEqual(result.channels, 2)
        XCTAssertGreaterThan(result.duration, 0)
    }
    
    func testAvailableSpeakersExposesBuiltinVoicesWhenModelsAreAvailable() async throws {
        let downloader = ModelDownloader()
        let ttsDir = await downloader.ttsModelDir(for: .mossTTSNano100M)
        let audioTokenizerDir = await downloader.tokenizerModelDir(for: .mossTTSNano100M)
        let paths = ModelPaths(ttsModelDir: ttsDir, audioTokenizerDir: audioTokenizerDir)
        
        guard paths.availability().isComplete else {
            throw XCTSkip("MOSS-TTS model files are not available in the default cache")
        }
        
        let tts = try await MOSSTTSKit(
            ttsModelDir: ttsDir,
            audioTokenizerDir: audioTokenizerDir,
            options: MOSSTTSOptions(maxLength: 3)
        )
        let availableSpeakers = await tts.availableSpeakers
        let builtinSpeakers = await tts.builtinSpeakers
        
        XCTAssertEqual(availableSpeakers.count, 18)
        XCTAssertEqual(builtinSpeakers.count, 18)
        XCTAssertEqual(availableSpeakers.first?.identifier, "Junhao")
        XCTAssertEqual(availableSpeakers.first?.displayName, "CN 欢迎关注模思智能")
    }
    
    func testSpeakProgressCallbackCanCancelWhenModelsAreAvailable() async throws {
        let downloader = ModelDownloader()
        let ttsDir = await downloader.ttsModelDir(for: .mossTTSNano100M)
        let audioTokenizerDir = await downloader.tokenizerModelDir(for: .mossTTSNano100M)
        let paths = ModelPaths(ttsModelDir: ttsDir, audioTokenizerDir: audioTokenizerDir)
        
        guard paths.availability().isComplete else {
            throw XCTSkip("MOSS-TTS model files are not available in the default cache")
        }
        
        let recorder = ProgressRecorder(cancelAfterStep: 2)
        let tts = try await MOSSTTSKit(
            ttsModelDir: ttsDir,
            audioTokenizerDir: audioTokenizerDir,
            options: MOSSTTSOptions(maxLength: 10, maxGeneratedFrames: 8)
        )
        let result = try await tts.speak(text: "你好") { progress in
            recorder.record(progress)
        }
        
        XCTAssertFalse(result.audioSamples.isEmpty)
        XCTAssertEqual(recorder.steps, [1, 2])
    }
    
    func testSpeakStreamProducesChunksWhenModelsAreAvailable() async throws {
        let downloader = ModelDownloader()
        let ttsDir = await downloader.ttsModelDir(for: .mossTTSNano100M)
        let audioTokenizerDir = await downloader.tokenizerModelDir(for: .mossTTSNano100M)
        let paths = ModelPaths(ttsModelDir: ttsDir, audioTokenizerDir: audioTokenizerDir)
        
        guard paths.availability().isComplete else {
            throw XCTSkip("MOSS-TTS model files are not available in the default cache")
        }
        
        let tts = try await MOSSTTSKit(
            ttsModelDir: ttsDir,
            audioTokenizerDir: audioTokenizerDir,
            options: MOSSTTSOptions(maxLength: 10, maxGeneratedFrames: 3)
        )
        let stream = try await tts.speakStream(text: "你好")
        
        var chunks: [MOSSTTSStreamChunk] = []
        for try await chunk in stream {
            chunks.append(chunk)
        }
        
        XCTAssertGreaterThanOrEqual(chunks.count, 2)
        XCTAssertTrue(chunks.dropLast().allSatisfy { !$0.newAudioSamples.isEmpty && !$0.isFinal })
        XCTAssertTrue(chunks.last?.isFinal == true)
        XCTAssertFalse(chunks.last?.audioSamples.isEmpty ?? true)
    }
}

private final class ProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private let cancelAfterStep: Int
    private var recordedSteps: [Int] = []
    
    init(cancelAfterStep: Int) {
        self.cancelAfterStep = cancelAfterStep
    }
    
    var steps: [Int] {
        lock.lock()
        defer { lock.unlock() }
        return recordedSteps
    }
    
    func record(_ progress: MOSSProgress) -> Bool {
        lock.lock()
        recordedSteps.append(progress.currentStep)
        lock.unlock()
        return progress.currentStep < cancelAfterStep
    }
}
