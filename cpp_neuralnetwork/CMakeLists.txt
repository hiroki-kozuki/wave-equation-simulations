cmake_minimum_required(VERSION 3.1)

project(main)

set(CMAKE_PREFIX_PATH "/mnt/c/Users/tjstu/Downloads/libtorch-shared-with-deps-2.5.1+cpu/libtorch")

include_directories("/mnt/c/Users/tjstu/Downloads/libtorch-shared-with-deps-2.5.1+cpu/libtorch/include/torch/csrc/api/include")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "Source file path: ${CMAKE_SOURCE_DIR}/main.cpp")

add_executable(main main.cpp 
                    include/network.h
                    )
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)