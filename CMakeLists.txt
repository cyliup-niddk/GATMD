cmake_minimum_required(VERSION 3.18)
project(autoencoder_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Accept either a standalone LibTorch root or PyTorch's CMake prefix. When
# neither is supplied, prefer the package belonging to the active Python.
if(DEFINED TORCH_DIR)
  list(PREPEND CMAKE_PREFIX_PATH "${TORCH_DIR}")
elseif(NOT Torch_DIR)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
      OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET RESULT_VARIABLE PYTORCH_QUERY_RESULT)
    if(PYTORCH_QUERY_RESULT EQUAL 0)
      list(PREPEND CMAKE_PREFIX_PATH "${PYTORCH_CMAKE_PREFIX}")
    endif()
  endif()
endif()

find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

add_executable(autoencoder_cpp main.cpp model.cpp)
add_executable(usage_cpp usage.cpp model.cpp)
target_link_libraries(autoencoder_cpp "${TORCH_LIBRARIES}" Threads::Threads)
target_link_libraries(usage_cpp "${TORCH_LIBRARIES}" Threads::Threads)
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(autoencoder_cpp PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_compile_options(usage_cpp PRIVATE ${TORCH_CXX_FLAGS_LIST})

include(CTest)
if(BUILD_TESTING)
  add_test(NAME autoencoder_integration
    COMMAND "${CMAKE_COMMAND}"
      -DEXECUTABLE=$<TARGET_FILE:autoencoder_cpp>
      -DUSAGE_EXECUTABLE=$<TARGET_FILE:usage_cpp>
      -DFIXTURE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/data/simple
      -DPEAKS_FILE=${CMAKE_CURRENT_SOURCE_DIR}/tests/peaks.csv
      -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/integration-output
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_test.cmake)
endif()
