// +build linux,amd64

package cntk

// #cgo LDFLAGS: -lCntk.Core-2.3
// #cgo LDFLAGS: -lCntk.Math-2.3
// #cgo LDFLAGS: -lCntk.PerformanceProfiler-2.3
// #cgo LDFLAGS: -lCntk.Eval-2.3
// #cgo LDFLAGS: -lmklml_intel -liomp5 -L${SRCDIR} -lstdc++
// #cgo LDFLAGS: -L/opt/cntk/cntk/lib -L/opt/cntk/cntk/dependencies/lib -L/opt/frameworks/cntk/cntk/lib -L/opt/frameworks/cntk/cntk/dependencies/lib
// #cgo CXXFLAGS: -std=c++11 -O3 -Wall -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function -Wno-reorder -Wno-unknown-pragmas
// #cgo CXXFLAGS: -I/usr/local/cuda/include -I/opt/cntk/Include -I/opt/frameworks/cntk/Include
// #cgo CXXFLAGS: -I${SRCDIR}/cbits -I${SRCDIR}/cbits/util -I${SRCDIR}/cuda
import "C"
