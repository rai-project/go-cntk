#ifndef ___PREDICT_HPP__
#define ___PREDICT_HPP__

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include "timer.h"

typedef void *PredictorContext;

PredictorContext NewCNTK(const char *modelFile, const char *deviceType,
                         const int deviceID);

void InitCNTK();

error_t PredictCNTK(PredictorContext pred, float *input,
                    const char *output_layer_name, int batch_size);

float *GetPredictionsCNTK(PredictorContext pred);

void DeleteCNTK(PredictorContext pred);

void StartProfilingCNTK(PredictorContext pred, const char *name,
                        const char *metadata);

void EndProfilingCNTK(PredictorContext pred);

void DisableProfilingCNTK(PredictorContext pred);

char *ReadProfileCNTK(PredictorContext pred);

int GetPredLenCNTK(PredictorContext pred);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // ___PREDICT_HPP__
