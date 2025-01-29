# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.30.5/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/divyanshdusad/Desktop/yoloV8arduino/YOLO_OpenPose_Project/openpose

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/divyanshdusad/Desktop/yoloV8arduino/build

# Utility rule file for openpose_lib.

# Include any custom commands dependencies for this target.
include CMakeFiles/openpose_lib.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/openpose_lib.dir/progress.make

CMakeFiles/openpose_lib: CMakeFiles/openpose_lib-complete

CMakeFiles/openpose_lib-complete: caffe/src/openpose_lib-stamp/openpose_lib-install
CMakeFiles/openpose_lib-complete: caffe/src/openpose_lib-stamp/openpose_lib-mkdir
CMakeFiles/openpose_lib-complete: caffe/src/openpose_lib-stamp/openpose_lib-download
CMakeFiles/openpose_lib-complete: caffe/src/openpose_lib-stamp/openpose_lib-update
CMakeFiles/openpose_lib-complete: caffe/src/openpose_lib-stamp/openpose_lib-patch
CMakeFiles/openpose_lib-complete: caffe/src/openpose_lib-stamp/openpose_lib-configure
CMakeFiles/openpose_lib-complete: caffe/src/openpose_lib-stamp/openpose_lib-build
CMakeFiles/openpose_lib-complete: caffe/src/openpose_lib-stamp/openpose_lib-install
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/Users/divyanshdusad/Desktop/yoloV8arduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'openpose_lib'"
	/opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E make_directory /Users/divyanshdusad/Desktop/yoloV8arduino/build/CMakeFiles
	/opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E touch /Users/divyanshdusad/Desktop/yoloV8arduino/build/CMakeFiles/openpose_lib-complete
	/opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E touch /Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-stamp/openpose_lib-done

caffe/src/openpose_lib-stamp/openpose_lib-build: caffe/src/openpose_lib-stamp/openpose_lib-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/Users/divyanshdusad/Desktop/yoloV8arduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Performing build step for 'openpose_lib'"
	cd /Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-build && $(MAKE)
	cd /Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-build && /opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E touch /Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-stamp/openpose_lib-build

caffe/src/openpose_lib-stamp/openpose_lib-configure: caffe/tmp/openpose_lib-cfgcmd.txt
caffe/src/openpose_lib-stamp/openpose_lib-configure: caffe/src/openpose_lib-stamp/openpose_lib-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/Users/divyanshdusad/Desktop/yoloV8arduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Performing configure step for 'openpose_lib'"
	cd /Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-build && /opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -DCMAKE_INSTALL_PREFIX:PATH=/Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe -DCMAKE_TOOLCHAIN_FILE= -DUSE_CUDNN=OFF -DCUDA_ARCH_NAME= -DCUDA_ARCH_BIN= -DCUDA_ARCH_PTX= -DCPU_ONLY=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_docs=OFF -DBUILD_python=OFF -DBUILD_python_layer=OFF -DUSE_LEVELDB=OFF -DUSE_LMDB=OFF -DUSE_OPENCV=OFF "-GUnix Makefiles" -S /Users/divyanshdusad/Desktop/yoloV8arduino/YOLO_OpenPose_Project/openpose/3rdparty/caffe -B /Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-build
	cd /Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-build && /opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E touch /Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-stamp/openpose_lib-configure

caffe/src/openpose_lib-stamp/openpose_lib-download: caffe/src/openpose_lib-stamp/openpose_lib-source_dirinfo.txt
caffe/src/openpose_lib-stamp/openpose_lib-download: caffe/src/openpose_lib-stamp/openpose_lib-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/Users/divyanshdusad/Desktop/yoloV8arduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "No download step for 'openpose_lib'"
	/opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E echo_append
	/opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E touch /Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-stamp/openpose_lib-download

caffe/src/openpose_lib-stamp/openpose_lib-install: caffe/src/openpose_lib-stamp/openpose_lib-build
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/Users/divyanshdusad/Desktop/yoloV8arduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Performing install step for 'openpose_lib'"
	cd /Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-build && $(MAKE) install
	cd /Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-build && /opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E touch /Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-stamp/openpose_lib-install

caffe/src/openpose_lib-stamp/openpose_lib-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/Users/divyanshdusad/Desktop/yoloV8arduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Creating directories for 'openpose_lib'"
	/opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -Dcfgdir= -P /Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/tmp/openpose_lib-mkdirs.cmake
	/opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E touch /Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-stamp/openpose_lib-mkdir

caffe/src/openpose_lib-stamp/openpose_lib-patch: caffe/src/openpose_lib-stamp/openpose_lib-patch-info.txt
caffe/src/openpose_lib-stamp/openpose_lib-patch: caffe/src/openpose_lib-stamp/openpose_lib-update
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/Users/divyanshdusad/Desktop/yoloV8arduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No patch step for 'openpose_lib'"
	/opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E echo_append
	/opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E touch /Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-stamp/openpose_lib-patch

caffe/src/openpose_lib-stamp/openpose_lib-update: caffe/src/openpose_lib-stamp/openpose_lib-update-info.txt
caffe/src/openpose_lib-stamp/openpose_lib-update: caffe/src/openpose_lib-stamp/openpose_lib-download
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/Users/divyanshdusad/Desktop/yoloV8arduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No update step for 'openpose_lib'"
	/opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E echo_append
	/opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E touch /Users/divyanshdusad/Desktop/yoloV8arduino/build/caffe/src/openpose_lib-stamp/openpose_lib-update

openpose_lib: CMakeFiles/openpose_lib
openpose_lib: CMakeFiles/openpose_lib-complete
openpose_lib: caffe/src/openpose_lib-stamp/openpose_lib-build
openpose_lib: caffe/src/openpose_lib-stamp/openpose_lib-configure
openpose_lib: caffe/src/openpose_lib-stamp/openpose_lib-download
openpose_lib: caffe/src/openpose_lib-stamp/openpose_lib-install
openpose_lib: caffe/src/openpose_lib-stamp/openpose_lib-mkdir
openpose_lib: caffe/src/openpose_lib-stamp/openpose_lib-patch
openpose_lib: caffe/src/openpose_lib-stamp/openpose_lib-update
openpose_lib: CMakeFiles/openpose_lib.dir/build.make
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold "Rerunning cmake after building Caffe submodule"
	/opt/homebrew/Cellar/cmake/3.30.5/bin/cmake /Users/divyanshdusad/Desktop/yoloV8arduino/YOLO_OpenPose_Project/openpose
	$(MAKE)
.PHONY : openpose_lib

# Rule to build all files generated by this target.
CMakeFiles/openpose_lib.dir/build: openpose_lib
.PHONY : CMakeFiles/openpose_lib.dir/build

CMakeFiles/openpose_lib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/openpose_lib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/openpose_lib.dir/clean

CMakeFiles/openpose_lib.dir/depend:
	cd /Users/divyanshdusad/Desktop/yoloV8arduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/divyanshdusad/Desktop/yoloV8arduino/YOLO_OpenPose_Project/openpose /Users/divyanshdusad/Desktop/yoloV8arduino/YOLO_OpenPose_Project/openpose /Users/divyanshdusad/Desktop/yoloV8arduino/build /Users/divyanshdusad/Desktop/yoloV8arduino/build /Users/divyanshdusad/Desktop/yoloV8arduino/build/CMakeFiles/openpose_lib.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/openpose_lib.dir/depend

