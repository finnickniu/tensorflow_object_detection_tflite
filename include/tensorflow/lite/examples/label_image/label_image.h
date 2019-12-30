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

#ifndef TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_LABEL_IMAGE_H_
#define TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_LABEL_IMAGE_H_
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"



namespace tflite {
namespace label_image {

struct Settings {
  bool verbose = false;
  bool accel = false;
  bool old_accel = false;
  bool input_floating = false;
  bool profiling = false;
  bool allow_fp16 = false;
  bool gl_backend = false;
  int loop_count = 1;
  float input_mean = 127.5f;
  float input_std = 127.5f;
  string model_name = "/home/apptech/tensorflow/c++/mobilenet_v1_1.0_224.tflite";
  tflite::FlatBufferModel* model;
  string input_bmp_name = "/home/apptech/tensorflow/c++/testdata/2.bmp";
  string labels_file_name = "./labels.txt";
  string input_layer_type = "uint8_t";
  int number_of_threads = 4;
  int number_of_results = 5;
  int max_profiling_buffer_entries = 1024;
  int number_of_warmup_runs = 2;
};

}  // namespace label_image
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_LABEL_IMAGE_H_
