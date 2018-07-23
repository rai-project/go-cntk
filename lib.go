// +build linux,amd64

package cntk

// #cgo LDFLAGS: -lcublas -lcudart -lcudnn -lcurand -lcusparse
// #cgo LDFLAGS: -lCntk.Core-2.5.1
// #cgo LDFLAGS: -lCntk.Math-2.5.1
// #cgo LDFLAGS: -lCntk.PerformanceProfiler-2.5.1
// #cgo LDFLAGS: -lCntk.Eval-2.5.1
// #cgo LDFLAGS: -lmklml_intel -liomp5 -L${SRCDIR} -lstdc++
// #cgo LDFLAGS: -L/home/as29/my_cntk/src/cntk/build/release/lib -L/home/as29/my_mklml/mklml/mklml_lnx_2018.0.3.20180406/lib -L/home/as29/my_cntk/cntk/cntk/cntk/dependencies/lib -L/usr/local/cuda/lib64
// #cgo CXXFLAGS: -std=c++11 -O3 -Wall -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function -Wno-reorder -Wno-unknown-pragmas
// #cgo CXXFLAGS: -I/usr/local/cuda/include -I/opt/cntk/Include -I/opt/frameworks/cntk/Include -I/home/as29/my_cntk/cntk/cntk/Include
// #cgo CXXFLAGS: -I${SRCDIR}/cbits -I${SRCDIR}/cbits/util -I${SRCDIR}/cuda
import "C"
