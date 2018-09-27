#ifdef __linux__

//#define _GLIBCXX_USE_CXX11_ABI 0 // see https://stackoverflow.com/a/33395489

#include "predict.hpp"
#include "json.hpp"
#include "timer.h"
#include "timer.impl.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

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

inline std::string wstrtostr(const std::wstring &wstr) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  return converter.to_bytes(wstr);
}

PredictorContext NewCNTK(const char *modelFile, const char *deviceType,
                         const int deviceID) {
  try {
    auto device = DeviceDescriptor::CPUDevice();
    if (deviceType != nullptr && std::string(deviceType) == "GPU") {
      //std::cerr << "cntk is using the gpu!!\n";
      device = DeviceDescriptor::GPUDevice(deviceID);
    }
    auto modelFunc =
        Function::Load(strtowstr(modelFile), device, ModelFormat::CNTKv2);
    Predictor *pred = new Predictor(modelFunc, device);
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
                        const char *output_layer_name, const int batchSize) {

  try {
    auto predictor = (Predictor *)pred;
    if (predictor == nullptr) {
      std::cerr << "CNTK prediction error on " << __LINE__ << "\n";
      return nullptr;
    }
    auto modelFunc = predictor->modelFunc_;

    const auto device = predictor->device_;

    // Get input variable. The model has only one single input.
    Variable inputVar = modelFunc->Arguments()[0];

    Variable outputVar;
    if (modelFunc->Outputs().size() == 1) {
      outputVar = modelFunc->Output();
    } else {
      const auto outputs = modelFunc->Outputs();
      const auto output_layer_name_string = strtowstr(output_layer_name);
      auto f = std::find_if(
          outputs.begin(), outputs.end(), [=](const Variable &var) {
            if (var.Name() == output_layer_name_string && var.IsOutput()) {
              return true;
            }
            return false;
          });
      if (f == outputs.end()) {
        std::cerr << "cannot find " << std::string(output_layer_name)
                  << " in the model. Valid outputs are: \n";
        for (const auto out : modelFunc->Outputs()) {
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
                                                    batchSize);
    ValuePtr inputVal = Value::CreateBatch(inputVar.Shape(), inputData, device);
    std::unordered_map<Variable, ValuePtr> inputDataMap = {
        {inputVar, inputVal}};

    // Create output data map. Using null as Value to indicate using system
    // allocated memory.
    // Alternatively, create a Value object and add it to the data map.
    std::unordered_map<Variable, ValuePtr> outputDataMap = {
        {outputVar, nullptr}};

    // Start evaluation on the device
    modelFunc->Evaluate(inputDataMap, outputDataMap, device);

    std::vector<std::vector<float>> resultsWrapper;

    CNTK::ValuePtr outputVal = outputDataMap[outputVar];
    outputVal.get()->CopyVariableValueTo(outputVar, resultsWrapper);
    const auto output_size = resultsWrapper[0].size();

  for (int ii =0 ;ii<10;ii++) {
    std::unordered_map<Variable, ValuePtr> outputDataMap = {
        {outputVar, nullptr}};
    modelFunc->Evaluate(inputDataMap, outputDataMap, device);

    std::vector<std::vector<float>> resultsWrapper;

    CNTK::ValuePtr outputVal = outputDataMap[outputVar];
    outputVal.get()->CopyVariableValueTo(outputVar, resultsWrapper);
  }
  const auto iters = 100;
  double es = 0.0;
  for (int ii =0 ;ii<iters;ii++) {
    std::unordered_map<Variable, ValuePtr> outputDataMap = {
        {outputVar, nullptr}};
    const auto start = now();
    modelFunc->Evaluate(inputDataMap, outputDataMap, device);
    cudaDeviceSynchronize();

    const auto end = now();
    es += elapsed_time(start, end);
    std::vector<std::vector<float>> resultsWrapper;

    CNTK::ValuePtr outputVal = outputDataMap[outputVar];
    outputVal.get()->CopyVariableValueTo(outputVar, resultsWrapper);
  }
 // std::cout << batchSize << ",";
  std::cout <<  es / iters << "\n";
    json preds = json::array();

    for (int cnt = 0; cnt < batchSize; cnt++) {
      for (int idx = 0; idx < output_size; idx++) {
        preds.push_back(
            {{"index", idx}, {"probability", resultsWrapper[cnt][idx]}});
      }
    }

    auto res = strdup(preds.dump().c_str());
    return res;
  } catch (const std::runtime_error &e) {
    std::cerr << "failed to perform predict on cntk model :: runtime error :: "
              << e.what() << "\n";
    return nullptr;
  } catch (const std::exception &e) {
    std::cerr << "failed to perform predict on cntk model :: exception :: "
              << e.what() << "\n";
    return nullptr;
  }
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
