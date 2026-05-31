import AVFoundation
import Foundation

/// 语音克隆参考音频的预处理策略。
public enum MOSSReferenceAudioProcessing: Sendable, Equatable {
    /// 自动选择一段连续、清晰、长度适中的人声片段作为 clone prompt。
    case automatic

    /// 不做片段选择，直接使用调用方传入的音频。
    case none
}

enum ReferenceAudioSegmentSelector {
    struct Configuration {
        var targetDuration: TimeInterval = 3.0
        var maxDuration: TimeInterval = 3.2
        var minDuration: TimeInterval = 1.5
        var windowDuration: TimeInterval = 0.05
        var maxSpeechGapDuration: TimeInterval = 0.20
        var leadingPaddingDuration: TimeInterval = 0.05
        var trailingPaddingDuration: TimeInterval = 0.25
        var minimumSpeechRegionDuration: TimeInterval = 0.50
        var absoluteRMSThreshold: Float = 0.008
        var relativeRMSThresholdRatio: Float = 0.12
    }

    static func prepareReferenceAudio(
        at url: URL,
        maxScanDuration: TimeInterval?,
        configuration: Configuration = Configuration()
    ) throws -> URL? {
        let loadedAudio = try loadMonoAudio(at: url, maxDuration: maxScanDuration)
        guard let range = selectBestSegmentRange(
            samples: loadedAudio.samples,
            sampleRate: loadedAudio.sampleRate,
            configuration: configuration
        ) else {
            return nil
        }

        let selectedSamples = Array(loadedAudio.samples[range])
        let temporaryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mosstts-reference-\(UUID().uuidString).wav")
        try writeMonoWAV(samples: selectedSamples, sampleRate: loadedAudio.sampleRate, to: temporaryURL)
        return temporaryURL
    }

    static func selectBestSegmentRange(
        samples: [Float],
        sampleRate: Int,
        configuration: Configuration = Configuration()
    ) -> Range<Int>? {
        guard sampleRate > 0, !samples.isEmpty else { return nil }

        let minSampleCount = Int(configuration.minDuration * Double(sampleRate))
        let maxSampleCount = Int(configuration.maxDuration * Double(sampleRate))
        let targetSampleCount = Int(configuration.targetDuration * Double(sampleRate))
        guard samples.count > minSampleCount else { return nil }

        let windowSize = max(1, Int(configuration.windowDuration * Double(sampleRate)))
        let rmsValues = stride(from: 0, to: samples.count, by: windowSize).map { startIndex -> Float in
            let endIndex = min(startIndex + windowSize, samples.count)
            let frameCount = max(1, endIndex - startIndex)
            var sum: Float = 0
            for sample in samples[startIndex..<endIndex] {
                sum += sample * sample
            }
            return sqrt(sum / Float(frameCount))
        }

        guard let maxRMS = rmsValues.max(), maxRMS > 0 else { return nil }
        let threshold = max(configuration.absoluteRMSThreshold, maxRMS * configuration.relativeRMSThresholdRatio)
        var speechWindows = rmsValues.map { $0 >= threshold }
        bridgeShortGaps(in: &speechWindows, maxGapWindows: max(1, Int(configuration.maxSpeechGapDuration / configuration.windowDuration)))

        let regions = speechRegions(
            from: speechWindows,
            windowSize: windowSize,
            sampleCount: samples.count,
            minimumSampleCount: Int(configuration.minimumSpeechRegionDuration * Double(sampleRate))
        )
        guard let bestRegion = bestRegion(regions, rmsValues: rmsValues, windowSize: windowSize, targetSampleCount: targetSampleCount) else {
            return nil
        }

        let leadingPadding = Int(configuration.leadingPaddingDuration * Double(sampleRate))
        let trailingPadding = Int(configuration.trailingPaddingDuration * Double(sampleRate))
        var start = max(0, bestRegion.lowerBound - leadingPadding)
        var end = min(samples.count, bestRegion.upperBound + trailingPadding)

        if end - start > maxSampleCount {
            start = bestRegion.lowerBound
            end = min(samples.count, start + targetSampleCount)
        }

        if end - start < minSampleCount {
            let missingSamples = minSampleCount - (end - start)
            let prepend = min(start, missingSamples / 2)
            start -= prepend
            end = min(samples.count, end + (missingSamples - prepend))
        }

        guard end > start else { return nil }
        if start == 0, end == samples.count, samples.count <= maxSampleCount {
            return nil
        }
        return start..<end
    }

    private static func bridgeShortGaps(in speechWindows: inout [Bool], maxGapWindows: Int) {
        var index = 0
        while index < speechWindows.count {
            guard !speechWindows[index] else {
                index += 1
                continue
            }

            let gapStart = index
            while index < speechWindows.count, !speechWindows[index] {
                index += 1
            }
            let gapEnd = index

            let hasSpeechBefore = gapStart > 0 && speechWindows[gapStart - 1]
            let hasSpeechAfter = gapEnd < speechWindows.count && speechWindows[gapEnd]
            if hasSpeechBefore, hasSpeechAfter, gapEnd - gapStart <= maxGapWindows {
                for gapIndex in gapStart..<gapEnd {
                    speechWindows[gapIndex] = true
                }
            }
        }
    }

    private static func speechRegions(
        from speechWindows: [Bool],
        windowSize: Int,
        sampleCount: Int,
        minimumSampleCount: Int
    ) -> [Range<Int>] {
        var regions: [Range<Int>] = []
        var index = 0

        while index < speechWindows.count {
            guard speechWindows[index] else {
                index += 1
                continue
            }

            let startWindow = index
            while index < speechWindows.count, speechWindows[index] {
                index += 1
            }

            let start = startWindow * windowSize
            let end = min(sampleCount, index * windowSize)
            if end - start >= minimumSampleCount {
                regions.append(start..<end)
            }
        }

        return regions
    }

    private static func bestRegion(
        _ regions: [Range<Int>],
        rmsValues: [Float],
        windowSize: Int,
        targetSampleCount: Int
    ) -> Range<Int>? {
        regions.max { left, right in
            score(region: left, rmsValues: rmsValues, windowSize: windowSize, targetSampleCount: targetSampleCount)
                < score(region: right, rmsValues: rmsValues, windowSize: windowSize, targetSampleCount: targetSampleCount)
        }
    }

    private static func score(
        region: Range<Int>,
        rmsValues: [Float],
        windowSize: Int,
        targetSampleCount: Int
    ) -> Float {
        let startWindow = max(0, region.lowerBound / windowSize)
        let endWindow = min(rmsValues.count, max(startWindow + 1, Int(ceil(Double(region.upperBound) / Double(windowSize)))))
        let averageRMS = rmsValues[startWindow..<endWindow].reduce(Float(0), +) / Float(max(1, endWindow - startWindow))
        let usableLength = Float(min(region.count, targetSampleCount)) / Float(max(1, targetSampleCount))
        let startPenalty = Float(region.lowerBound) / Float(max(1, targetSampleCount)) * 0.02
        return usableLength + averageRMS - startPenalty
    }

    private static func loadMonoAudio(at url: URL, maxDuration: TimeInterval?) throws -> (samples: [Float], sampleRate: Int) {
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat
        let requestedFrameCount: AVAudioFrameCount
        if let maxDuration, maxDuration > 0 {
            requestedFrameCount = min(
                AVAudioFrameCount(audioFile.length),
                AVAudioFrameCount((maxDuration * format.sampleRate).rounded(.down))
            )
        } else {
            requestedFrameCount = AVAudioFrameCount(audioFile.length)
        }

        guard requestedFrameCount > 0 else {
            throw MOSSTTSError.invalidInput("Reference audio is empty")
        }

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: requestedFrameCount) else {
            throw MOSSTTSError.audioProcessingFailed("Failed to create reference audio buffer")
        }
        try audioFile.read(into: buffer)

        guard let channelData = buffer.floatChannelData else {
            throw MOSSTTSError.audioProcessingFailed("Failed to read reference audio samples")
        }

        let channels = Int(format.channelCount)
        let frames = Int(buffer.frameLength)
        var samples: [Float] = []
        samples.reserveCapacity(frames)

        for frame in 0..<frames {
            var sum: Float = 0
            for channel in 0..<channels {
                sum += channelData[channel][frame]
            }
            samples.append(sum / Float(channels))
        }

        return (samples, Int(format.sampleRate))
    }

    private static func writeMonoWAV(samples: [Float], sampleRate: Int, to url: URL) throws {
        guard let format = AVAudioFormat(
            standardFormatWithSampleRate: Double(sampleRate),
            channels: 1
        ) else {
            throw MOSSTTSError.audioProcessingFailed("Failed to create reference audio format")
        }

        let audioFile = try AVAudioFile(
            forWriting: url,
            settings: format.settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            throw MOSSTTSError.audioProcessingFailed("Failed to create selected reference audio buffer")
        }

        buffer.frameLength = AVAudioFrameCount(samples.count)
        guard let channelData = buffer.floatChannelData else {
            throw MOSSTTSError.audioProcessingFailed("Failed to write selected reference audio samples")
        }

        for index in samples.indices {
            channelData[0][index] = samples[index]
        }

        try audioFile.write(from: buffer)
    }
}
