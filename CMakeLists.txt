cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(demo_1)


set(CMAKE_CXX_STANDARD 17)

set(REQUIRED_LIBS ${TFLITE_LIBS})
# list(APPEND REQUIRED_LIBS ${TFLITE_LIBS})

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_library(TFLITE_LIBRARY tensorflowlite "lib")
#find_library(TFLITE_LIBRARY tensorflow-lite "lib")
find_library(FLATBUFFER_LIBRARY flatbuffers "lib")
include_directories("include")

set(REQUIRED_LIBS "stdc++fs")
list(APPEND REQUIRED_LIBS ${TFLITE_LIBRARY})
list(APPEND REQUIRED_LIBS ${FLATBUFFER_LIBRARY})
list(APPEND REQUIRED_LIBS ${OpenCV_LIBS})
list(APPEND REQUIRED_LIBS ${Threads_LIBS})

set(CMAKE_C_FLAGS "-pthread")
set(SOURCE_FILES demo.cpp)

add_executable("demo" ${SOURCE_FILES})
SET_TARGET_PROPERTIES(demo PROPERTIES LINK_FLAGS "-no-pie")
target_link_libraries("demo" ${REQUIRED_LIBS} -lpthread)

