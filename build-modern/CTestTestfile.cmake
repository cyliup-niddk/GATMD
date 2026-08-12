# CMake generated Testfile for 
# Source directory: /run/media/liveuser/SANDI_LARGE/autoencoder/cpp
# Build directory: /run/media/liveuser/SANDI_LARGE/autoencoder/cpp/build-modern
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(autoencoder_integration "/usr/bin/cmake" "-DEXECUTABLE=/run/media/liveuser/SANDI_LARGE/autoencoder/cpp/build-modern/autoencoder_cpp" "-DFIXTURE_DIR=/run/media/liveuser/SANDI_LARGE/autoencoder/cpp/tests/data" "-DOUTPUT_DIR=/run/media/liveuser/SANDI_LARGE/autoencoder/cpp/build-modern/integration-output" "-P" "/run/media/liveuser/SANDI_LARGE/autoencoder/cpp/tests/integration_test.cmake")
set_tests_properties(autoencoder_integration PROPERTIES  _BACKTRACE_TRIPLES "/run/media/liveuser/SANDI_LARGE/autoencoder/cpp/CMakeLists.txt;34;add_test;/run/media/liveuser/SANDI_LARGE/autoencoder/cpp/CMakeLists.txt;0;")
