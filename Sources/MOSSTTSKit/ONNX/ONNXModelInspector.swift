import Foundation

/// ONNX model input/output metadata extracted directly from ModelProto bytes.
public struct ONNXModelIOInfo: Sendable, Equatable {
    public let modelURL: URL
    public let inputs: [ONNXModelTensorInfo]
    public let outputs: [ONNXModelTensorInfo]
    
    public init(modelURL: URL, inputs: [ONNXModelTensorInfo], outputs: [ONNXModelTensorInfo]) {
        self.modelURL = modelURL
        self.inputs = inputs
        self.outputs = outputs
    }
    
    public var markdownDescription: String {
        func lines(_ title: String, _ tensors: [ONNXModelTensorInfo]) -> [String] {
            var result = ["### \(title)"]
            if tensors.isEmpty {
                result.append("- None")
            } else {
                result.append(contentsOf: tensors.map { "- \($0.markdownDescription)" })
            }
            return result
        }
        
        return (["## \(modelURL.lastPathComponent)"] + lines("Inputs", inputs) + lines("Outputs", outputs))
            .joined(separator: "\n")
    }
}

/// Tensor metadata from an ONNX ValueInfoProto.
public struct ONNXModelTensorInfo: Sendable, Equatable {
    public let name: String
    public let elementType: ONNXModelElementType
    public let dimensions: [ONNXModelDimension]
    
    public init(name: String, elementType: ONNXModelElementType, dimensions: [ONNXModelDimension]) {
        self.name = name
        self.elementType = elementType
        self.dimensions = dimensions
    }
    
    public var shapeDescription: String {
        "[" + dimensions.map(\.description).joined(separator: ", ") + "]"
    }
    
    public var markdownDescription: String {
        "`\(name)` \(elementType.description) \(shapeDescription)"
    }
}

/// ONNX tensor dimension, either static, symbolic, or unknown.
public enum ONNXModelDimension: Sendable, Equatable, CustomStringConvertible {
    case value(Int64)
    case parameter(String)
    case unknown
    
    public var description: String {
        switch self {
        case .value(let value):
            return "\(value)"
        case .parameter(let parameter):
            return parameter
        case .unknown:
            return "?"
        }
    }
}

/// ONNX TensorProto.DataType.
public enum ONNXModelElementType: Int, Sendable, Equatable, CustomStringConvertible {
    case undefined = 0
    case float = 1
    case uint8 = 2
    case int8 = 3
    case uint16 = 4
    case int16 = 5
    case int32 = 6
    case int64 = 7
    case string = 8
    case bool = 9
    case float16 = 10
    case double = 11
    case uint32 = 12
    case uint64 = 13
    case complex64 = 14
    case complex128 = 15
    case bfloat16 = 16
    case float8E4M3FN = 17
    case float8E4M3FNUZ = 18
    case float8E5M2 = 19
    case float8E5M2FNUZ = 20
    case uint4 = 21
    case int4 = 22
    case float4E2M1 = 23
    
    public var description: String {
        switch self {
        case .undefined: return "undefined"
        case .float: return "float32"
        case .uint8: return "uint8"
        case .int8: return "int8"
        case .uint16: return "uint16"
        case .int16: return "int16"
        case .int32: return "int32"
        case .int64: return "int64"
        case .string: return "string"
        case .bool: return "bool"
        case .float16: return "float16"
        case .double: return "float64"
        case .uint32: return "uint32"
        case .uint64: return "uint64"
        case .complex64: return "complex64"
        case .complex128: return "complex128"
        case .bfloat16: return "bfloat16"
        case .float8E4M3FN: return "float8e4m3fn"
        case .float8E4M3FNUZ: return "float8e4m3fnuz"
        case .float8E5M2: return "float8e5m2"
        case .float8E5M2FNUZ: return "float8e5m2fnuz"
        case .uint4: return "uint4"
        case .int4: return "int4"
        case .float4E2M1: return "float4e2m1"
        }
    }
    
    public static func fromONNXRawValue(_ rawValue: Int) -> ONNXModelElementType {
        Self(rawValue: rawValue) ?? .undefined
    }
}

/// Lightweight ONNX ModelProto metadata reader.
public enum ONNXModelInspector {
    public static func inspect(modelURL: URL) throws -> ONNXModelIOInfo {
        let data = try Data(contentsOf: modelURL, options: [.mappedIfSafe])
        let graph = try modelGraph(from: data)
        let initializerNames = Set(graph.initializers.map(\.name))
        let inputs = graph.inputs.filter { !initializerNames.contains($0.name) }
        return ONNXModelIOInfo(modelURL: modelURL, inputs: inputs, outputs: graph.outputs)
    }
    
    public static func inspect(modelPaths: ModelPaths) throws -> [ONNXModelIOInfo] {
        let modelURLs = ModelDownloader.ttsModelFiles
            .filter { $0.hasSuffix(".onnx") }
            .map(modelPaths.ttsFile)
            + ModelDownloader.audioTokenizerFiles
            .filter { $0.hasSuffix(".onnx") }
            .map(modelPaths.tokenizerFile)
        
        return try modelURLs
            .filter { FileManager.default.fileExists(atPath: $0.path) }
            .map(inspect(modelURL:))
    }
    
    public static func markdownReport(for modelPaths: ModelPaths) throws -> String {
        let models = try inspect(modelPaths: modelPaths)
        if models.isEmpty {
            return "No ONNX model files found."
        }
        return models.map(\.markdownDescription).joined(separator: "\n\n")
    }
    
    private static func modelGraph(from modelData: Data) throws -> ParsedGraph {
        var reader = ProtobufReader(data: modelData)
        while let field = try reader.nextField() {
            if field.number == 7, case .lengthDelimited(let data) = field.value {
                return try parseGraph(data)
            }
        }
        throw MOSSTTSError.modelLoadFailed("ONNX ModelProto graph field not found")
    }
    
    private static func parseGraph(_ data: Data) throws -> ParsedGraph {
        var reader = ProtobufReader(data: data)
        var inputs: [ONNXModelTensorInfo] = []
        var outputs: [ONNXModelTensorInfo] = []
        var initializers: [ParsedInitializer] = []
        
        while let field = try reader.nextField() {
            switch (field.number, field.value) {
            case (5, .lengthDelimited(let data)):
                initializers.append(try parseInitializer(data))
            case (11, .lengthDelimited(let data)):
                inputs.append(try parseValueInfo(data))
            case (12, .lengthDelimited(let data)):
                outputs.append(try parseValueInfo(data))
            default:
                continue
            }
        }
        
        return ParsedGraph(inputs: inputs, outputs: outputs, initializers: initializers)
    }
    
    private static func parseInitializer(_ data: Data) throws -> ParsedInitializer {
        var reader = ProtobufReader(data: data)
        var name = ""
        
        while let field = try reader.nextField() {
            if field.number == 1, case .lengthDelimited(let data) = field.value {
                name = String(data: data, encoding: .utf8) ?? ""
            }
        }
        
        return ParsedInitializer(name: name)
    }
    
    private static func parseValueInfo(_ data: Data) throws -> ONNXModelTensorInfo {
        var reader = ProtobufReader(data: data)
        var name = ""
        var elementType: ONNXModelElementType = .undefined
        var dimensions: [ONNXModelDimension] = []
        
        while let field = try reader.nextField() {
            switch (field.number, field.value) {
            case (1, .lengthDelimited(let data)):
                name = String(data: data, encoding: .utf8) ?? ""
            case (2, .lengthDelimited(let data)):
                let tensorType = try parseType(data)
                elementType = tensorType.elementType
                dimensions = tensorType.dimensions
            default:
                continue
            }
        }
        
        return ONNXModelTensorInfo(name: name, elementType: elementType, dimensions: dimensions)
    }
    
    private static func parseType(_ data: Data) throws -> ParsedTensorType {
        var reader = ProtobufReader(data: data)
        while let field = try reader.nextField() {
            if field.number == 1, case .lengthDelimited(let data) = field.value {
                return try parseTensorType(data)
            }
        }
        return ParsedTensorType(elementType: .undefined, dimensions: [])
    }
    
    private static func parseTensorType(_ data: Data) throws -> ParsedTensorType {
        var reader = ProtobufReader(data: data)
        var elementType: ONNXModelElementType = .undefined
        var dimensions: [ONNXModelDimension] = []
        
        while let field = try reader.nextField() {
            switch (field.number, field.value) {
            case (1, .varint(let value)):
                elementType = ONNXModelElementType.fromONNXRawValue(Int(value))
            case (2, .lengthDelimited(let data)):
                dimensions = try parseTensorShape(data)
            default:
                continue
            }
        }
        
        return ParsedTensorType(elementType: elementType, dimensions: dimensions)
    }
    
    private static func parseTensorShape(_ data: Data) throws -> [ONNXModelDimension] {
        var reader = ProtobufReader(data: data)
        var dimensions: [ONNXModelDimension] = []
        
        while let field = try reader.nextField() {
            if field.number == 1, case .lengthDelimited(let data) = field.value {
                dimensions.append(try parseDimension(data))
            }
        }
        
        return dimensions
    }
    
    private static func parseDimension(_ data: Data) throws -> ONNXModelDimension {
        var reader = ProtobufReader(data: data)
        var dimension: ONNXModelDimension = .unknown
        
        while let field = try reader.nextField() {
            switch (field.number, field.value) {
            case (1, .varint(let value)):
                dimension = .value(Int64(value))
            case (2, .lengthDelimited(let data)):
                dimension = .parameter(String(data: data, encoding: .utf8) ?? "?")
            default:
                continue
            }
        }
        
        return dimension
    }
}

private struct ParsedGraph {
    let inputs: [ONNXModelTensorInfo]
    let outputs: [ONNXModelTensorInfo]
    let initializers: [ParsedInitializer]
}

private struct ParsedInitializer {
    let name: String
}

private struct ParsedTensorType {
    let elementType: ONNXModelElementType
    let dimensions: [ONNXModelDimension]
}

private struct ProtobufReader {
    enum WireValue {
        case varint(UInt64)
        case fixed64
        case lengthDelimited(Data)
        case fixed32
    }
    
    struct Field {
        let number: Int
        let value: WireValue
    }
    
    private let bytes: [UInt8]
    private var offset = 0
    
    init(data: Data) {
        self.bytes = Array(data)
    }
    
    mutating func nextField() throws -> Field? {
        guard offset < bytes.count else { return nil }
        
        let key = try readVarint()
        let fieldNumber = Int(key >> 3)
        let wireType = Int(key & 0x7)
        
        switch wireType {
        case 0:
            return Field(number: fieldNumber, value: .varint(try readVarint()))
        case 1:
            try skip(count: 8)
            return Field(number: fieldNumber, value: .fixed64)
        case 2:
            let length = Int(try readVarint())
            let data = try readData(count: length)
            return Field(number: fieldNumber, value: .lengthDelimited(data))
        case 5:
            try skip(count: 4)
            return Field(number: fieldNumber, value: .fixed32)
        default:
            throw MOSSTTSError.modelLoadFailed("Unsupported protobuf wire type: \(wireType)")
        }
    }
    
    private mutating func readVarint() throws -> UInt64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0
        
        while offset < bytes.count {
            let byte = bytes[offset]
            offset += 1
            
            result |= UInt64(byte & 0x7F) << shift
            if byte & 0x80 == 0 {
                return result
            }
            
            shift += 7
            if shift >= 64 {
                throw MOSSTTSError.modelLoadFailed("Invalid protobuf varint")
            }
        }
        
        throw MOSSTTSError.modelLoadFailed("Unexpected end of protobuf varint")
    }
    
    private mutating func readData(count: Int) throws -> Data {
        guard count >= 0, offset + count <= bytes.count else {
            throw MOSSTTSError.modelLoadFailed("Unexpected end of protobuf length-delimited field")
        }
        
        let data = Data(bytes[offset..<offset + count])
        offset += count
        return data
    }
    
    private mutating func skip(count: Int) throws {
        guard count >= 0, offset + count <= bytes.count else {
            throw MOSSTTSError.modelLoadFailed("Unexpected end of protobuf fixed field")
        }
        offset += count
    }
}
