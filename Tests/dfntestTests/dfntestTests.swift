import XCTest
@testable import dfntest

final class dfntestTests: XCTestCase {
    func testExample() {
        let q: DeviceQueue = CpuQueue()
        let m = Matrix<Float>()
        var result = Matrix<Float>()
        q.sum(x: m, alongAxis: -1, result: &result)
        q.softmax(m, result: &result)
        
//        let left = Matrix<Float>()
//        let right = Matrix<Float>()

//        let retainedAdd = q.createAdd(lhs: left)
//
//        let sum = retainedAdd.execute(left, right)
        
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}

// user device protocol extensions
//==============================================================================
// written in Swift and compiled by MLIR
extension DeviceQueue {
    func userFunction1<T>(x: T) -> T where
        T: TensorView, T.Element: FloatingPoint {
            print("this is userFunction1 written in Swift " +
                "and compiled by MLIR")
            return x
    }

    func userFunction2<T>(lhs: T, rhs: T) -> T where
        T: TensorView, T.Element: FloatingPoint {
            print("this is userFunction2 written in Swift " +
                "and compiled by MLIR")
            return lhs
    }
}

//------------------------------------------------------------------------------
// user device queue class, device specific extensions
extension CpuQueue {
    func userFunction2<T>(lhs: T, rhs: T) -> T where
        T: TensorView, T.Element: FloatingPoint {
            print("this is overriden userFunction2" +
                "calling out to an external library like BLAS")
            return lhs
    }
}

extension CudaQueue {
    func userFunction1<T>(x: T) -> T where
        T: TensorView, T.Element: FloatingPoint {
            print("this is overriden userFunction1" +
                "calling out to cuDNN or a handwritten .cu file")
            return x
    }
}
