# Description:
# TensorFlow Lite Example Label Image.

load("//tensorflow/lite:build_def.bzl", "tflite_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(glob([
    "testdata/*.bmp",
]))

cc_binary(
    name = "object_detection",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "object_detection.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":bitmap_helpers",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/profiling:profiler",
        "//tensorflow/lite/tools/evaluation:utils",
        "@com_google_absl//absl/memory",
    ] + select({
        "//tensorflow:android": ["//tensorflow/lite/delegates/gpu:gl_delegate"],
        "//tensorflow:android_arm64": ["//tensorflow/lite/delegates/gpu:gl_delegate"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "bitmap_helpers",
    srcs = ["bitmap_helpers.cc"],
    hdrs = [
        "bitmap_helpers.h",
        "bitmap_helpers_impl.h",
        "object_detection.h",
    ],
    deps = [
        "//tensorflow/lite:builtin_op_data",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
    ] + select({
        "//tensorflow:android": ["//tensorflow/lite/delegates/gpu:gl_delegate"],
        "//tensorflow:android_arm64": ["//tensorflow/lite/delegates/gpu:gl_delegate"],
        "//conditions:default": [],
    }),
)

cc_library(

    name = "opencv",

    srcs = glob(["lib/x86_64-linux-gnu/*.so*"]),

    hdrs =  glob([ "include/opencv2/*.h", "include/opencv2/*.hpp", ]),

    includes = ["include"],

    visibility = ["//visibility:public"], 

    linkstatic = 1,

)

cc_test(
    name = "object_detection_test",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "object_detection.cc",
    ],
    data = [
        "testdata/grace_hopper.bmp",
    ],
    deps = [
        ":bitmap_helpers",
        "@com_google_googletest//:gtest",
    ],
)