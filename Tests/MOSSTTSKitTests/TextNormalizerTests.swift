import XCTest
@testable import MOSSTTSKit

final class TextNormalizerTests: XCTestCase {
    private let normalizer = TextNormalizer()

    func testNormalizesChineseEllipsisToSentenceBoundary() {
        let text = """
        利娜正睡在我身边，她的双手握着，就像她平常睡觉时那样……

        我一点都不想再睡了。
        """

        let normalized = normalizer.normalize(text)

        XCTAssertFalse(normalized.contains("…"))
        XCTAssertEqual(normalized, "利娜正睡在我身边，她的双手握着，就像她平常睡觉时那样。 我一点都不想再睡了。")
    }

    func testNormalizesAsciiEllipsisToSentenceBoundary() {
        XCTAssertEqual(
            normalizer.normalize("I woke up... then looked around."),
            "I woke up. then looked around."
        )
    }

    func testNormalizesDanglingSpeakerLabelColon() {
        XCTAssertEqual(normalizer.normalize("Taiguanglin："), "Taiguanglin.")
        XCTAssertEqual(normalizer.normalize("旁白："), "旁白。")
    }

    func testRemovesChineseQuotationMarks() {
        XCTAssertEqual(
            normalizer.normalize("“米歇，准确地讲，是在一百三十五万年以前。”"),
            "米歇，准确地讲，是在一百三十五万年以前。"
        )
    }

    func testNormalizesQuotedParagraphBoundaries() {
        let text = """
        当我们重新坐好之后，涛就开始了她那奇怪的故事。
        “米歇，准确地讲，是在一百三十五万年以前。”
        """

        XCTAssertEqual(
            normalizer.normalize(text),
            "当我们重新坐好之后，涛就开始了她那奇怪的故事。 米歇，准确地讲，是在一百三十五万年以前。"
        )
    }

    func testNormalizesAsciiDashSeparatorsToSentenceBoundary() {
        XCTAssertEqual(
            normalizer.normalize("非常值得一读，---揭示地球史前文明。"),
            "非常值得一读。 揭示地球史前文明。"
        )
    }

    func testNormalizesChineseEmDashSeparatorsToSentenceBoundary() {
        XCTAssertEqual(
            normalizer.normalize("他们有眼却不看，有耳却不闻。——《圣经》"),
            "他们有眼却不看，有耳却不闻。 《圣经》"
        )
    }

    func testKeepsSingleHyphensInWords() {
        XCTAssertEqual(
            normalizer.normalize("MOSS-TTS-Nano is local-first."),
            "MOSS-TTS-Nano is local-first."
        )
    }

    func testLineBreaksBecomeSentenceBoundaries() {
        XCTAssertEqual(
            normalizer.normalize("第一行\n第二行"),
            "第一行。 第二行"
        )
    }

    func testKeepsExistingSentenceTerminators() {
        XCTAssertEqual(
            normalizer.normalize("第一句。\n第二句？"),
            "第一句。 第二句？"
        )
    }
}
