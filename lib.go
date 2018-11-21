// +build linux,amd64

package cntk

// #cgo CXXFLAGS: -I${SRCDIR}/cbits -I/opt/cntk/Include -std=c++11 -O3 -Wall -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function -Wno-reorder -Wno-unknown-pragmas
// #cgo LDFLAGS: -lmklml_intel -liomp5 -L${SRCDIR} -lstdc++
// #cgo LDFLAGS: -L/opt/cntk/cntk/lib -L/opt/cntk/cntk/dependencies/lib
// #cgo LDFLAGS: -lCntk.Core-2.6 -lCntk.Math-2.6 -lCntk.PerformanceProfiler-2.6 -lCntk.Eval-2.6
// #cgo !nogpu CXXFLAGS: -I/usr/local/cuda/include
// #cgo !nogpu LDFLAGS: -lcublas -lcudart -lcudnn -lcurand -lcusparse -L/usr/local/cuda/lib64
import "C"
