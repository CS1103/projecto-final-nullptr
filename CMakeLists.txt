cmake_minimum_required(VERSION 3.20)
project(projecto_final_nullptr LANGUAGES CXX)

# --- estandar de c++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- biblioteca con tu código real ------------
file(GLOB_RECURSE CORE_SRC src/**/*.cpp)
add_library(utec_core ${CORE_SRC})
target_include_directories(utec_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

# --- ejecutable principal --------------------
add_executable(main_app main.cpp)
target_link_libraries(main_app PRIVATE utec_core)

# --- tests con catch2 -------------------
enable_testing()

file(GLOB TEST_SOURCES ${CMAKE_SOURCE_DIR}/tests/*.cpp)
add_executable(tests_runner ${TEST_SOURCES})
target_include_directories(tests_runner
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tools/catch
)
target_link_libraries(tests_runner PRIVATE utec_core)
add_test(NAME all_tests COMMAND tests_runner)
set_target_properties(tests_runner PROPERTIES LINK_FLAGS "-mconsole")