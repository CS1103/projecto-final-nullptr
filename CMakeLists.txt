cmake_minimum_required(VERSION 3.29)
project(projecto_final_nullptr LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

# --- biblioteca con tu código real ------------
file(GLOB_RECURSE CORE_SRC src/**/*.cpp)
add_library(utec_core ${CORE_SRC})
target_include_directories(utec_core PUBLIC include)

# --- ejecutable principal --------------------
add_executable(main_app main.cpp)
target_link_libraries(main_app PRIVATE utec_core)

# --- ejecutable de pruebas -------------------
# add_executable(TextLoaderTest tests/text_loader_test.cpp)
# target_link_libraries(TextLoaderTest PRIVATE utec_core)

# --- soporte para testing --------------------
# enable_testing()
# add_test(NAME TextLoaderTestRun COMMAND TextLoaderTest)
# add_test(NAME NeuralNetworkTestRun COMMAND NeuralNetworkTest)
# add_test(NAME DatasetUtilsTestRun COMMAND DatasetUtilsTest)
