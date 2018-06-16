// +build linux,amd64

package cntk

// #cgo LDFLAGS: -lCntk.Core-2.5.1
// #cgo LDFLAGS: -lCntk.Math-2.5.1
// #cgo LDFLAGS: -lCntk.PerformanceProfiler-2.5.1
// #cgo LDFLAGS: -lCntk.Eval-2.5.1
// #cgo LDFLAGS: -lmklml_intel -liomp5 -L${SRCDIR} -lstdc++
// #cgo LDFLAGS: -L/home/as29/my_cntk/cntk/cntk/lib -L/home/as29/my_cntk/cntk/cntk/dependencies/lib
// #cgo CXXFLAGS: -std=c++11 -O3 -Wall -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function -Wno-reorder -Wno-unknown-pragmas
// #cgo CXXFLAGS: -I/usr/local/cuda/include -I/home/as29/my_cntk/cntk/Include 
// #cgo CXXFLAGS: -I${SRCDIR}/cbits -I${SRCDIR}/cbits/util -I${SRCDIR}/cuda
import "C"
