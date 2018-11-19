// +build linux,!ppc64le

package cntk

// #include <stdlib.h>
// #include "cbits/predict.hpp"
import "C"
import (
	"context"
	"unsafe"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework/framework/options"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
)

type Predictor struct {
	ctx     C.PredictorContext
	options *options.Options
}

func New(ctx context.Context, opts0 ...options.Option) (*Predictor, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_new")
	defer span.Finish()

	opts := options.New(opts0...)
	modelFile := string(opts.Graph())
	if !com.IsFile(modelFile) {
		return nil, errors.Errorf("file %s not found", modelFile)
	}

	modelFileString := C.CString(modelFile)
	defer C.free(unsafe.Pointer(modelFileString))

	deviceType := "CPU"
	deviceId := 0
	if opts.UsesGPU() {
		if !nvidiasmi.HasGPU {
			return nil, errors.New("no GPU device")
		}
		deviceType = "GPU"
		for _, d := range opts.Devices() {
			if d.Type() == options.CUDA_DEVICE {
				deviceId = d.ID()
				break
			}
		}
	}

	deviceTypeString := C.CString(deviceType)
	defer C.free(unsafe.Pointer(deviceTypeString))

	ctx := C.NewCNTK(
		modelFileString,
		deviceTypeString,
		C.int(deviceId),
	)
	return &Predictor{
		ctx:     ctx,
		options: opts,
	}, nil
}

func prod(arry []uint32) int64 {
	accum := int64(1)
	for _, e := range arry {
		accum *= int64(e)
	}
	return accum
}

func (p *Predictor) Predict(ctx context.Context, data []float32, outputLayerName0 string, shape []uint32) error {
	if outputLayerName0 == "" {
		return nil, errors.New("expecting a valid (non-empty) output layer name")
	}
	outputLayerName := C.CString(outputLayerName0)
	defer C.free(unsafe.Pointer(outputLayerName))

	batchSize := p.options.BatchSize()
	dataLen := len(data)
	shapeLen := prod(shape)

	inputCount := dataLen / shapeLen
	if batchSize > inputCount {
		padding := make([]float32, (batchSize-inputCount)*shapeLen)
		data = append(data, padding...)
	}

	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")
	defer span.Finish()

	ptr := (*C.float)(unsafe.Pointer(&input[0]))
	r := C.PredictCNTK(p.ctx, ptr, outputLayerName, C.int(batchSize))
	if r == nil {
		return errors.New("failed to perform CNTK prediction")
	}
	defer C.free(unsafe.Pointer(r))

	return nil
}

func (p *Predictor) ReadPredictionOutput(ctx context.Context) ([]float32, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_read_prediction_output")
	defer span.Finish()

	batchSize := p.options.BatchSize()
	predLen := int(C.GetPredLenCNTK(p.ctx))
	length := batchSize * predLen

	cPredictions := C.GetPredictionsCNTK(p.ctx)
	if cPredictions == nil {
		return nil, errors.New("empty predictions")
	}

	slice := (*[1 << 30]float32)(unsafe.Pointer(cPredictions))[:length:length]

	return slice, nil
}

func (p Predictor) Close() {
	C.DeleteCNTK(p.ctx)
}

func (p *Predictor) StartProfiling(name, metadata string) error {
	cname := C.CString(name)
	cmetadata := C.CString(metadata)
	defer C.free(unsafe.Pointer(cname))
	defer C.free(unsafe.Pointer(cmetadata))
	C.StartProfilingCNTK(p.ctx, cname, cmetadata)
	return nil
}

func (p *Predictor) EndProfiling() error {
	C.EndProfilingCNTK(p.ctx)
	return nil
}

func (p *Predictor) DisableProfiling() error {
	C.DisableProfilingCNTK(p.ctx)
	return nil
}

func (p *Predictor) ReadProfile() (string, error) {
	cstr := C.ReadProfileCNTK(p.ctx)
	if cstr == nil {
		return "", errors.New("failed to read nil profile")
	}
	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr), nil
}
