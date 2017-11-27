#ifdef __linux__

#define _GLIBCXX_USE_CXX11_ABI 0 // see https://stackoverflow.com/a/33395489

#include "predict.hpp"
#include "json.hpp"
#include "timer.h"
#include "timer.impl.hpp"

#include <algorithm>
#include <iostream>
#include <vector>
#include <locale>
#include <codecvt>
#include <string>

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

    if (modelFunc_) {
      // modelFunc_->destroy();
    }
    if (prof_) {
      prof_->reset();
      delete prof_;
      prof_ = nullptr;
    }
  }

  FunctionPtr modelFunc_;
  profile *prof_{nullptr};
  bool prof_registered_{false};
};


inline std::wstring strtowstr(const std::string& str)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    return converter.from_bytes(str);
}


PredictorContext NewCNTK(const char *modelFile, int batch, const char*deviceType, const int deviceID) {
  try {
     auto device = DeviceDescriptor::CPUDevice();


    auto modelFunc = Function::Load(strtowstr(modelFile), device, ModelFormat::CNTKv2);
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
                            const char *input_layer_name,
                            const char *output_layer_name,
                            const int batchSize) {

  auto predictor = (Predictor *)pred;

  if (predictor == nullptr) {
    std::cerr << "CNTK prediction error on " << __LINE__ << "\n";
    return nullptr;
  }
  auto modelFunc = predictor->modelFunc_;
  return nullptr;
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
