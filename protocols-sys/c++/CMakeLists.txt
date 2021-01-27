cmake_minimum_required(VERSION 3.10)
project(DELPHI VERSION 1.0 LANGUAGES C CXX)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(SEAL_DIR ${CMAKE_SOURCE_DIR}/SEAL/native/src/cmake)

set(LIB "src/lib")
set(BIN "src/bin")

include_directories(${CMAKE_SOURCE_DIR}/src/lib)

add_library( DelphiOffline STATIC
    ${LIB}/interface.cpp ${LIB}/conv2d.cpp ${LIB}/fc_layer.cpp
    ${LIB}/triples_gen.cpp)

find_package(SEAL 3.4.5 EXACT REQUIRED)
find_package(OpenMP REQUIRED)
target_link_libraries(DelphiOffline SEAL::seal OpenMP::OpenMP_CXX)


option(UNITTESTS "Build unittests" OFF)
if(UNITTESTS)
    include_directories(${CMAKE_SOURCE_DIR}/src/bin)
    add_executable(unittests ${BIN}/unittests.cpp ${LIB}/im2col.cpp)
    add_executable(benchmark ${BIN}/benchmark.cpp ${LIB}/im2col.cpp)
    add_executable(interface ${BIN}/interface.c)

    # Unittests make use of Eigen
    find_package (Eigen3 3.3 REQUIRED NO_MODULE)
    target_link_libraries (unittests Eigen3::Eigen)
    target_link_libraries (benchmark Eigen3::Eigen)

    target_link_libraries (benchmark DelphiOffline)
    target_link_libraries (unittests DelphiOffline)
    target_link_libraries (interface DelphiOffline)
endif()

install(TARGETS DelphiOffline DESTINATION .)
