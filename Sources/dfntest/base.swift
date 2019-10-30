//******************************************************************************
// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


// ********
// THIS IS A MINIMAL CONTEXTURAL MOCKUP USING Softmax as an example
// This file contains definitions that would be in a base distribution
// of the product
// ********

//------------------------------------------------------------------------------
// tensor
public protocol TensorView {
    // the unconstrained element type stored in the collection
    associatedtype Element
}

public struct Matrix<Element>: TensorView {
    // dummy
    public let data = [Element]()
}

//------------------------------------------------------------------------------
// on device memory buffer
public protocol DeviceArray: class { }
public class CpuDeviceArray: DeviceArray { }
public class CudaDeviceArray: DeviceArray { }

//------------------------------------------------------------------------------
// Device operator lowest level intrinsic protocol
public protocol DeviceIntrinsics {
    /// Adds two tensors producing their sum
    func add<T>(lhs: T, rhs: T, result: inout T)
        where T: TensorView, T.Element: Numeric
    /// Computes the element-wise `exp`
    func exp<T>(x: T, result: inout T) where
        T: TensorView, T.Element: FloatingPoint
    /// sums the input along the axis
    func sum<T>(x: T, alongAxis axis: Int, result: inout T) where
        T: TensorView, T.Element: Numeric

    
    //<<<<<<<<<<<<<<<<<<<
    func createConv<T, FN>(x: T) -> FN where
        T: TensorView, T.Element: Numeric,
        FN: RetainedConvProtocol
}

//------------------------------------------------------------------------------
// DeviceIntrinsics default implementation
// *** NOTE: The default implementation is written in Swift and is device
// agnostic!
public extension DeviceIntrinsics {
    /// Adds two tensors producing their sum
    func add<T>(lhs: T, rhs: T, result: inout T)
        where T: TensorView, T.Element: Numeric {
            
    }
    
    /// Computes the element-wise `exp`
    func exp<T>(x: T, result: inout T) where
        T: TensorView, T.Element: FloatingPoint {
            
    }
    
    /// sums the input along the axis
    func sum<T>(x: T, alongAxis axis: Int, result: inout T) where
        T: TensorView, T.Element: Numeric {
            
    }

    //<<<<<<<<<<<<<<<<<<<
    // I want to return different generic implementations of RetainedConv
    func createConv<T, FN>(x: T) -> FN where
        T: TensorView, T.Element: Numeric,
        FN: RetainedConvProtocol
    {
        return RetainedConv(x: x)
    }
}

//------------------------------------------------------------------------------
// Device operator higher level protocols
public protocol DeviceDeepLearning {
    func softmax<T>(_ x: T, result: inout T) where
        T: TensorView, T.Element: FloatingPoint
}

// The corresponding gradient functions are separated
public protocol _vjpDeviceDeepLearning: DeviceDeepLearning {
    func _vjpSoftmax<T>(_ x: T) -> (T, (T) -> T) where
        T: TensorView, T.Element: FloatingPoint
}

//------------------------------------------------------------------------------
// DeviceDeepLearning default implementation
// *** NOTE: The default implementation is written in Swift and is device
// agnostic!
public extension DeviceDeepLearning {
    func softmax<T>(_ x: T, result: inout T) where
        T: TensorView, T.Element: FloatingPoint {
            
    }
}

public extension _vjpDeviceDeepLearning {
    func _vjpSoftmax<T>(_ x: T) -> (T, (T) -> T) where
        T: TensorView, T.Element: FloatingPoint
    {
        fatalError()
//        let value = softmax(x)
//        return (value, { v in
//            let sumChannels = sum(x: (v * value), alongAxes: -1)
//            return (v - sumChannels) * value
//        })
    }
}

//------------------------------------------------------------------------------
// Device queue base protocol
public protocol DeviceQueue:
    class,
    DeviceIntrinsics,
    DeviceDeepLearning
{
    // a function implemented in the application space
    func createArray(byteCount: Int) -> DeviceArray
}

//------------------------------------------------------------------------------
// Cpu Device queue implementation
public class CpuQueue: DeviceQueue {
    // a function implemented in the application space
    public func createArray(byteCount: Int) -> DeviceArray {
        return CpuDeviceArray()
    }
}

//------------------------------------------------------------------------------
// Cpu Device queue implementation
public class CudaQueue: DeviceQueue {
    // a function implemented in the application space
    public func createArray(byteCount: Int) -> DeviceArray {
        return CudaDeviceArray()
    }
}
