# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/divyanshdusad/Desktop/yoloV8arduino/YOLO_OpenPose_Project/openpose/3rdparty/caffe")
  file(MAKE_DIRECTORY "/Users/divyanshdusad/Desktop/yoloV8arduino/YOLO_OpenPose_Project/openpose/3rdparty/caffe")
endif()
file(MAKE_DIRECTORY
  "/Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-build"
  "/Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe"
  "/Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/tmp"
  "/Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-stamp"
  "/Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src"
  "/Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-stamp${cfgdir}") # cfgdir has leading slash
endif()
