// +build linux,amd64

package cntk

// #cgo LDFLAGS: -lcublas -lcudart -lcudnn -lcurand -lcusparse
// #cgo LDFLAGS: -lCntk.Core-2.6
// #cgo LDFLAGS: -lCntk.Math-2.6
// #cgo LDFLAGS: -lCntk.PerformanceProfiler-2.6
// #cgo LDFLAGS: -lCntk.Eval-2.6
// #cgo LDFLAGS: -lmklml_intel -liomp5 -L${SRCDIR} -lstdc++
// #cgo LDFLAGS: -L/opt/cntk/cntk/lib -L/opt/cntk/cntk/dependencies/lib -L/usr/local/cuda/lib64
// #cgo CXXFLAGS: -std=c++11 -O3 -Wall -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function -Wno-reorder -Wno-unknown-pragmas
// #cgo CXXFLAGS: -I/usr/local/cuda/include -I/opt/cntk/Include
// #cgo CXXFLAGS: -I${SRCDIR}/cbits -I${SRCDIR}/cbits/util -I${SRCDIR}/cuda
import "C"
