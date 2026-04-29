import Foundation
import XCTest
@testable import MOSSTTSKit

final class ONNXModelInspectorTests: XCTestCase {
    func testInspectorParsesMinimalModelProto() throws {
        let model = Proto.field(7, .message(
            Proto.field(11, .message(valueInfo(name: "input_ids", elementType: 7, dims: [.symbol("batch"), .symbol("seq")]))) +
            Proto.field(11, .message(valueInfo(name: "weight", elementType: 1, dims: [.value(4), .value(4)]))) +
            Proto.field(12, .message(valueInfo(name: "logits", elementType: 1, dims: [.symbol("batch"), .symbol("seq"), .value(1024)]))) +
            Proto.field(5, .message(Proto.field(1, .string("weight"))))
        ))
        
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("onnx")
        try model.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        
        let info = try ONNXModelInspector.inspect(modelURL: url)
        
        XCTAssertEqual(info.inputs.count, 1)
        XCTAssertEqual(info.inputs[0].name, "input_ids")
        XCTAssertEqual(info.inputs[0].elementType, .int64)
        XCTAssertEqual(info.inputs[0].dimensions, [.parameter("batch"), .parameter("seq")])
        
        XCTAssertEqual(info.outputs.count, 1)
        XCTAssertEqual(info.outputs[0].name, "logits")
        XCTAssertEqual(info.outputs[0].elementType, .float)
        XCTAssertEqual(info.outputs[0].dimensions, [.parameter("batch"), .parameter("seq"), .value(1024)])
        XCTAssertTrue(info.markdownDescription.contains("`input_ids` int64 [batch, seq]"))
        XCTAssertTrue(info.markdownDescription.contains("`logits` float32 [batch, seq, 1024]"))
    }
    
    private func valueInfo(name: String, elementType: Int, dims: [TestDimension]) -> Data {
        Proto.field(1, .string(name)) +
        Proto.field(2, .message(
            Proto.field(1, .message(
                Proto.field(1, .varint(UInt64(elementType))) +
                Proto.field(2, .message(shape(dims)))
            ))
        ))
    }
    
    private func shape(_ dims: [TestDimension]) -> Data {
        dims.reduce(Data()) { data, dim in
            data + Proto.field(1, .message(dimension(dim)))
        }
    }
    
    private func dimension(_ dim: TestDimension) -> Data {
        switch dim {
        case .value(let value):
            return Proto.field(1, .varint(UInt64(value)))
        case .symbol(let symbol):
            return Proto.field(2, .string(symbol))
        }
    }
}

private enum TestDimension {
    case value(Int)
    case symbol(String)
}

private enum Proto {
    enum Value {
        case varint(UInt64)
        case string(String)
        case message(Data)
    }
    
    static func field(_ number: Int, _ value: Value) -> Data {
        switch value {
        case .varint(let value):
            return varint(UInt64(number << 3)) + varint(value)
        case .string(let string):
            let data = Data(string.utf8)
            return varint(UInt64((number << 3) | 2)) + varint(UInt64(data.count)) + data
        case .message(let data):
            return varint(UInt64((number << 3) | 2)) + varint(UInt64(data.count)) + data
        }
    }
    
    static func varint(_ value: UInt64) -> Data {
        var value = value
        var data = Data()
        
        while value >= 0x80 {
            data.append(UInt8(value & 0x7F) | 0x80)
            value >>= 7
        }
        
        data.append(UInt8(value))
        return data
    }
}
