import XCTest
@testable import MOSSTTSKit

final class ReferenceAudioSegmentSelectorTests: XCTestCase {
    func testSelectsFirstContinuousSpeechRegionAndAvoidsSecondUtterance() throws {
        let sampleRate = 24_000
        let samples = makeSamples(
            sampleRate: sampleRate,
            segments: [
                (duration: 2.75, amplitude: 0.20),
                (duration: 0.75, amplitude: 0.0003),
                (duration: 2.25, amplitude: 0.20),
                (duration: 0.25, amplitude: 0.0003)
            ]
        )

        let range = ReferenceAudioSegmentSelector.selectBestSegmentRange(
            samples: samples,
            sampleRate: sampleRate
        )

        XCTAssertNotNil(range)
        let selected = try XCTUnwrap(range)
        XCTAssertEqual(selected.lowerBound, 0, accuracy: Int(0.10 * Double(sampleRate)))
        XCTAssertGreaterThanOrEqual(Double(selected.count) / Double(sampleRate), 2.8)
        XCTAssertLessThanOrEqual(Double(selected.count) / Double(sampleRate), 3.2)
        XCTAssertLessThan(selected.upperBound, Int(3.3 * Double(sampleRate)))
    }

    func testLeavesShortCleanReferenceAudioUnchanged() {
        let sampleRate = 24_000
        let samples = makeSamples(
            sampleRate: sampleRate,
            segments: [
                (duration: 2.2, amplitude: 0.20)
            ]
        )

        let range = ReferenceAudioSegmentSelector.selectBestSegmentRange(
            samples: samples,
            sampleRate: sampleRate
        )

        XCTAssertNil(range)
    }

    private func makeSamples(
        sampleRate: Int,
        segments: [(duration: TimeInterval, amplitude: Float)]
    ) -> [Float] {
        var samples: [Float] = []
        for segment in segments {
            let count = Int(segment.duration * Double(sampleRate))
            samples.append(contentsOf: repeatElement(segment.amplitude, count: count))
        }
        return samples
    }
}
