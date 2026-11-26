import Foundation

#if os(macOS)
import PenGPTModelImport
#endif

@main
struct PenGPTConvert {
    static func main() {
        #if os(macOS)
        var arguments = Array(CommandLine.arguments.dropFirst())
        let isStyle = arguments.first == "--style"
        if isStyle { arguments.removeFirst() }
        var component: NumPyModel.Component?
        if !isStyle, arguments.first == "--component" {
            guard arguments.count == 4, let value = NumPyModel.Component(rawValue: arguments[1]) else {
                usage()
            }
            component = value
            arguments.removeFirst(2)
        }
        guard arguments.count == 2 else { usage() }
        do {
            let input = URL(fileURLWithPath: arguments[0])
            let output = URL(fileURLWithPath: arguments[1])
            if isStyle {
                try NumPyModel.convertStyle(input, to: output)
            } else {
                try NumPyModel.convert(input, to: output, component: component)
            }
        } catch {
            FileHandle.standardError.write(Data("pengpt-convert: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
        #else
        fatalError("Run pengpt-convert on macOS.")
        #endif
    }
    #if os(macOS)
    private static func usage() -> Never {
        FileHandle.standardError.write(Data(
            "Usage: pengpt-convert [--component hand|sketch|words] INPUT.npz OUTPUT.pengpt\n       pengpt-convert --style INPUT.npy OUTPUT.json\n".utf8
        ))
        exit(2)
    }
    #endif

}
