#ifndef __CNTK_PREDICT_HPP__
#define __CNTK_PREDICT_HPP__

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef void *PredictorContext;

PredictorContext NewCNTK(const char *modelFile, int batch, const char*deviceType, const int deviceID);
void DeleteCNTK(PredictorContext pred);
const char *PredictCNTK(PredictorContext pred, float *input,
                            const char *input_layer_name,
                            const char *output_layer_name, const int batchSize);

void CNTKInit();

void CNTKStartProfiling(PredictorContext pred, const char *name,
                            const char *metadata);

void CNTKEndProfiling(PredictorContext pred);

void CNTKDisableProfiling(PredictorContext pred);

char *CNTKReadProfile(PredictorContext pred);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __CNTK_PREDICT_HPP__
