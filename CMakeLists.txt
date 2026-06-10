cmake_minimum_required(VERSION 3.16)
project(ai_webcam CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Homebrew installs onnxruntime config files under opt/onnxruntime
if(APPLE AND EXISTS "/opt/homebrew/opt/onnxruntime")
  list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/onnxruntime")
elseif(APPLE AND EXISTS "/usr/local/opt/onnxruntime")
  list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt/onnxruntime")
endif()

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE ${OpenCV_LIBS} onnxruntime::onnxruntime)
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
