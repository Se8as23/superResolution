cmake_minimum_required(VERSION 3.10)
project(SuperResONNX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ruta a ONNX Runtime
if(NOT DEFINED ONNXRUNTIME_DIR)
  message(FATAL_ERROR "Debe definir ONNXRUNTIME_DIR al ejecutar cmake")
endif()

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(principal principal.cpp)

target_link_libraries(principal ${OpenCV_LIBS} onnxruntime)

