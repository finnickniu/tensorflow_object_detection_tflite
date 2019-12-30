/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/examples/object_detection/object_detection.h"

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
// #include <opencv2/opencv.hpp>
#include "absl/memory/memory.h"
#include "tensorflow/lite/examples/object_detection/get_top_n.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/examples/object_detection/bitmap_helpers.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

#define LOG(x) std::cerr

namespace tflite {
namespace object_detection {

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;

TfLiteDelegatePtr CreateGPUDelegate(Settings* s) {
#if defined(__ANDROID__)
  TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
  gpu_opts.inference_preference =
      TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  gpu_opts.inference_priority1 =
      s->allow_fp16 ? TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY
                    : TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
  return evaluation::CreateGPUDelegate(s->model, &gpu_opts);
#else
  return evaluation::CreateGPUDelegate(s->model);
#endif
}

TfLiteDelegatePtrMap GetDelegates(Settings* s) {
  TfLiteDelegatePtrMap delegates;
  if (s->gl_backend) {
    auto delegate = CreateGPUDelegate(s);
    if (!delegate) {
      LOG(INFO) << "GPU acceleration is unsupported on this platform.";
    } else {
      delegates.emplace("GPU", std::move(delegate));
    }
  }


  return delegates;
}



void RunInference(Settings* s) {

  std:: cout << "run...." << std::endl;

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());
 
  s->model = model.get();

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  interpreter->AllocateTensors();

  interpreter->SetAllowFp16PrecisionForFp32(s->allow_fp16);
  interpreter->SetNumThreads(1);

  int image_width = 300;
  int image_height = 300;
  int image_channels = 3;
  std::vector<uint8_t> in = read_bmp(s->input_bmp_name, &image_width,
                                     &image_height, &image_channels, s);
  // cv::Mat image;
  // image = cv::imread(s->input_bmp_name);
  // cv::resize(image, image, cv::Size(),image_width,image_height);
  // std::vector<float> array(image.rows*image.cols);
  // if (image.isContinuous())
  //     array = image.data;
  int input = interpreter->inputs()[0];

  const std::vector<int> inputs = interpreter->inputs();
  interpreter->Invoke();


  auto delegates_ = GetDelegates(s);
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];


  switch (interpreter->tensor(input)->type) {
  case kTfLiteFloat32:
    s->input_floating = true;
    resize<float>(interpreter->typed_tensor<float>(input), in.data(),
                  image_height, image_width, image_channels, wanted_height,
                  wanted_width, wanted_channels, s);
    break;
  case kTfLiteUInt8:
    resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in.data(),
                    image_height, image_width, image_channels, wanted_height,
                    wanted_width, wanted_channels, s);
    break;
  } 

  auto profiler =
      absl::make_unique<profiling::Profiler>(s->max_profiling_buffer_entries);
  interpreter->SetProfiler(profiler.get());


  if (s->profiling) profiler->StartProfiling();

  if (s->profiling) {
  profiler->StopProfiling();
  auto profile_events = profiler->GetProfileEvents();
  for (int i = 0; i < profile_events.size(); i++) {
    auto subgraph_index = profile_events[i]->event_subgraph_index;
    auto op_index = profile_events[i]->event_metadata;
    const auto subgraph = interpreter->subgraph(subgraph_index);
    const auto node_and_registration =
        subgraph->node_and_registration(op_index);
    const TfLiteRegistration registration = node_and_registration->second;
    // PrintProfilingInfo(profile_events[i], subgraph_index, op_index,
    //                     registration);
    }
  }

  auto outputs = interpreter->typed_output_tensor<uint8_t>(0);

  auto output = interpreter->outputs()[0];

  // for (int i = 0; i < outputs.size();i++){
  //   auto output = interpreter->outputs()[i];
  std:: cout << "output:"<< outputs << std::endl;
  // }
//   const float threshold = 0.001f;
//   std::vector<std::pair<float, int>> top_results;
//   TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
//     // assume output dims to be something like (1, 1, ... ,size)
//   auto output_size = output_dims->data[output_dims->size - 1];

//   switch (interpreter->tensor(output)->type) {
//   case kTfLiteFloat32:
//     get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
//                       s->number_of_results, threshold, &top_results, true);
//     break;
//   case kTfLiteUInt8:
//     get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
//                         output_size, s->number_of_results, threshold,
//                         &top_results, false);
//     break;
//   }
// for (const auto& result : top_results) {
//   const float confidence = result.first;
//   const int index = result.second;
//   std:: cout << "confidence:"<< confidence << std::endl;
//   std:: cout << "index:"<< index << std::endl;
//   std:: cout << "result:"<< result << std::endl;


//   }
}

int Main(int argc, char** argv) {
  Settings s;


  RunInference(&s);
  return 0;
}

}  // namespace label_image
}  // namespace tflite

int main(int argc, char** argv) {
  return tflite::object_detection::Main(argc, argv);
}
