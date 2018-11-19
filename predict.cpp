#ifdef __linux__

//#define _GLIBCXX_USE_CXX11_ABI 0 // see https://stackoverflow.com/a/33395489

#include "predict.hpp"
#include "json.hpp"
#include "timer.h"
#include "timer.impl.hpp"

#include <algorithm>
#include <codecvt>
#include <iostream>
#include <locale>
#include <string>
#include <vector>

#include "CNTKLibrary.h"
#include "Eval.h"

using namespace CNTK;

using json = nlohmann::json;

#define CHECK(status)                                                          \
  {                                                                            \
    if (status != 0) {                                                         \
      std::cerr << "Cuda failure on line " << __LINE__                         \
                << " status =  " << status << "\n";                            \
      return nullptr;                                                          \
    }                                                                          \
  }

class Predictor {
public:
  Predictor(FunctionPtr modelFunc, DeviceDescriptor device)
      : modelFunc_(modelFunc), device_(device){};
  void Predict(float *input, const char *output_layer_name,
               const int batch_size);
  ~Predictor() {
    if (prof_) {
      prof_->reset();
      delete prof_;
      prof_ = nullptr;
    }
  }

  FunctionPtr modelFunc_{nullptr};
  DeviceDescriptor device_{DeviceDescriptor::CPUDevice()};
  int pred_len_;
  void *result_{nullptr};
  bool prof_enabled_{false};
  profile *prof_{nullptr};
};

inline std::wstring strtowstr(const std::string &str) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  return converter.from_bytes(str);
}

inline std::string wstrtostr(const std::wstring &wstr) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  return converter.to_bytes(wstr);
}

void Predictor::Predict(float *input, const char *output_layer_name,
                        const int batch_size) {
  if (result_ != nullptr) {
    free(result_);
    result_ = nullptr;
  }

  // Get input variable. The model has only one single input.
  Variable inputVar = modelFunc_->Arguments()[0];

  Variable outputVar;
  if (modelFunc_->Outputs().size() == 1) {
    outputVar = modelFunc_->Output();
  } else {
    const auto outputs = modelFunc_->Outputs();
    const auto output_layer_name_string = strtowstr(output_layer_name);
    auto f =
        std::find_if(outputs.begin(), outputs.end(), [=](const Variable &var) {
          if (var.Name() == output_layer_name_string && var.IsOutput()) {
            return true;
          }
          return false;
        });
    if (f == outputs.end()) {
      std::cerr << "cannot find " << std::string(output_layer_name)
                << " in the model. Valid outputs are: \n";
      for (const auto out : modelFunc_->Outputs()) {
        std::cerr << wstrtostr(out.AsString())
                  << " with name = " << wstrtostr(out.Name()) << "\n";
      }
      std::cerr << "make sure that the layer exists.";
      return nullptr;
    }
    outputVar = *f;
  }

  // Create input value and input data map
  std::vector<float> inputData(input, input + inputVar.Shape().TotalSize() *
                                                  batch_size);
  ValuePtr inputVal = Value::CreateBatch(inputVar.Shape(), inputData, device_);
  std::unordered_map<Variable, ValuePtr> inputDataMap = {{inputVar, inputVal}};

  // Create output data map. Using null as Value to indicate using system
  // allocated memory.
  // Alternatively, create a Value object and add it to the data map.
  std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

  // Start evaluation on the device
  modelFunc_->Evaluate(inputDataMap, outputDataMap, device_);

  std::vector<std::vector<float>> resultsWrapper;

  CNTK::ValuePtr outputVal = outputDataMap[outputVar];
  outputVal.get()->CopyVariableValueTo(outputVar, resultsWrapper);

  pred_len_ = resultsWrapper[0].size();
  const auto pred_size = pred_len_ * sizeof(float);
  std::vector<float> ret;
  for (int cnt = 0; cnt < batch_size; cnt++) {
    memcpy((float *)result_ + cnt * pred_size, &resultsWrapper[cnt], pred_size);
  }
}

PredictorContext NewCNTK(const char *modelFile, const char *deviceType,
                         const int deviceID) {
  try {
    auto device = DeviceDescriptor::CPUDevice();
    if (deviceType != nullptr && std::string(deviceType) == "GPU") {
      // std::cerr << "cntk is using the gpu!!\n";
      device = DeviceDescriptor::GPUDevice(deviceID);
    }
    auto modelFunc =
        Function::Load(strtowstr(modelFile), device, ModelFormat::CNTKv2);
    Predictor *pred = new Predictor(modelFunc, device);
    return (PredictorContext)pred;
  } catch (const std::invalid_argument &ex) {
    RuntimeError("exception:  %s\n", ex.what());
    errno = EINVAL;
    return nullptr;
  } catch (std::exception &ex) {
    RuntimeError("exception: catch all [  %s  ]\n", ex.what());
    return nullptr;
  }
}

void InitCNTK() {}

error_t PredictCNTK(PredictorContext pred, float *input,
                    const char *output_layer_name, const int batch_size) {
  try {
    auto predictor = (Predictor *)pred;
    if (predictor == nullptr) {
      std ::cout << __func__ << "  " << __LINE__ << " ... got a null pointer\n";
      return error_invalid_memory;
    }
    predictor->Predict(input, output_layer_name, batch_size);
    return success;
  } catch (std::exception &ex) {
    RuntimeError("exception: catch all [  %s  ]\n", ex.what());
    return error_exception;
  }
}

float *GetPredictionsCNTK(PredictorContext pred) {
  try {
    auto predictor = (Predictor *)pred;
    if (predictor == nullptr) {
      return nullptr;
    }
    if (predictor->result_ == nullptr) {
      throw std::runtime_error("expected a non-nil result");
    }
    return (float *)predictor->result_;
  } catch (std::exception &ex) {
    RuntimeError("exception: catch all [  %s  ]\n", ex.what());
    return nullptr;
  }
}

void DeleteCNTK(PredictorContext pred) {
  try {
    auto predictor = (Predictor *)pred;
    if (predictor == nullptr) {
      return;
    }
    if (predictor->result_) {
      free(predictor->result_);
    }
    if (predictor->prof_) {
      predictor->prof_->reset();
      delete predictor->prof_;
      predictor->prof_ = nullptr;
    }
    delete predictor;
  } catch (std::exception &ex) {
    RuntimeError("exception: catch all [  %s  ]\n", ex.what());
    return;
  }
}

void StartProfilingCNTK(PredictorContext pred, const char *name,
                        const char *metadata) {
  try {
    if (name == nullptr) {
      name = "";
    }
    if (metadata == nullptr) {
      metadata = "";
    }
    if (pred == nullptr) {
      return;
    }
    auto predictor = (Predictor *)pred;
    predictor->profile_enabled_ = true;
    if (predictor->prof_ == nullptr) {
      predictor->prof_ = new profile(name, metadata);
    } else {
      predictor->prof_->reset();
    }
  } catch (std::exception &ex) {
    RuntimeError("exception: catch all [  %s  ]\n", ex.what());
    return;
  }
}

void EndProfilingCNTK(PredictorContext pred) {
  try {
    auto predictor = (Predictor *)pred;
    if (predictor == nullptr) {
      return;
    }
    if (predictor->prof_) {
      predictor->prof_->end();
    }
  } catch (std::exception &ex) {
    RuntimeError("exception: catch all [  %s  ]\n", ex.what());
    return;
  }
}

void DisableProfilingCNTK(PredictorContext pred) {
  try {
    auto predictor = (Predictor *)pred;
    if (predictor == nullptr) {
      return;
    }
    if (predictor->prof_) {
      predictor->prof_->reset();
    }
  } catch (std::exception &ex) {
    RuntimeError("exception: catch all [  %s  ]\n", ex.what());
    return;
  }
}

char *ReadProfileCNTK(PredictorContext pred) {
  try {
    auto predictor = (Predictor *)pred;
    if (predictor == nullptr) {
      return strdup("");
    }
    if (predictor->prof_ == nullptr) {
      return strdup("");
    }
    const auto s = predictor->prof_->read();
    const auto cstr = s.c_str();
    return strdup(cstr);
  } catch (std::exception &ex) {
    RuntimeError("exception: catch all [  %s  ]\n", ex.what());
    return nullptr;
  }
}

int GetPredLenCNTK(PredictorContext pred) {
  try {
    auto predictor = (Predictor *)pred;
    if (predictor == nullptr) {
      return 0;
    }
    return predictor->pred_len_;
  } catch (std::exception &ex) {
    RuntimeError("exception: catch all [  %s  ]\n", ex.what());
    return 0;
  }
}

#endif // __linux__
