#ifdef __linux__

#define _GLIBCXX_USE_CXX11_ABI 0 // see https://stackoverflow.com/a/33395489

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
  Predictor(FunctionPtr modelFunc) : modelFunc_(modelFunc){};
  ~Predictor() {
    if (prof_) {
      prof_->reset();
      delete prof_;
      prof_ = nullptr;
    }
  }

  FunctionPtr modelFunc_{nullptr};
  DeviceDescriptor device_{DeviceDescriptor::CPUDevice()};
  profile *prof_{nullptr};
  bool prof_registered_{false};
};

inline std::wstring strtowstr(const std::string &str) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  return converter.from_bytes(str);
}

PredictorContext NewCNTK(const char *modelFile, int batch,
                         const char *deviceType, const int deviceID) {
  try {
    auto device = DeviceDescriptor::CPUDevice();

    auto modelFunc =
        Function::Load(strtowstr(modelFile), device, ModelFormat::CNTKv2);
    Predictor *pred = new Predictor(modelFunc);
    return (PredictorContext)pred;
  } catch (const std::invalid_argument &ex) {
    return nullptr;
  }
}

void DeleteCNTK(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  delete predictor;
}

const char *PredictCNTK(PredictorContext pred, float *input,
                        const int batchSize) {

  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    std::cerr << "CNTK prediction error on " << __LINE__ << "\n";
    return nullptr;
  }
  auto modelFunc = predictor->modelFunc_;

  const auto device = predictor->device_;

  // Get input variable. The model has only one single input.
  Variable inputVar = modelFunc->Arguments()[0];

  // The model has only one output.
  // If the model has more than one output, use modelFunc->Outputs to get the
  // list of output variables.
  Variable outputVar = modelFunc->Output();

  // Create input value and input data map
  std::vector<float> inputData(input, input + inputVar.Shape().TotalSize());
  ValuePtr inputVal = Value::CreateBatch(inputVar.Shape(), inputData, device);
  std::unordered_map<Variable, ValuePtr> inputDataMap = {{inputVar, inputVal}};

  // Create output data map. Using null as Value to indicate using system
  // allocated memory.
  // Alternatively, create a Value object and add it to the data map.
  std::unordered_map<Variable, ValuePtr> outputDataMap = {{outputVar, nullptr}};

  // Start evaluation on the device
  modelFunc->Evaluate(inputDataMap, outputDataMap, device);

  std::vector<std::vector<float>> resultsWrapper;

  CNTK::ValuePtr outputVal = outputDataMap[outputVar];
  outputVal.get()->CopyVariableValueTo(outputVar, resultsWrapper);
  auto output = resultsWrapper[0];

  json preds = json::array();

  for (int cnt = 0; cnt < batchSize; cnt++) {
    const auto output_size = output.size() / batchSize;
    for (int idx = 0; idx < output_size; idx++) {
      preds.push_back(
          {{"index", idx}, {"probability", output[cnt * output_size + idx]}});
    }
  }

  auto res = strdup(preds.dump().c_str());
  return res;
}

void CNTKInit() {}

void CNTKStartProfiling(PredictorContext pred, const char *name,
                        const char *metadata) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (name == nullptr) {
    name = "";
  }
  if (metadata == nullptr) {
    metadata = "";
  }
  if (predictor->prof_ == nullptr) {
    predictor->prof_ = new profile(name, metadata);
  } else {
    predictor->prof_->reset();
  }
}

void CNTKEndProfiling(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->end();
  }
}

void CNTKDisableProfiling(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->reset();
  }
}

char *CNTKReadProfile(PredictorContext pred) {
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
}

#endif // __linux__
