cmake_minimum_required(VERSION 3.12)
project(minitorch_c C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W3 /O2)
else()
    add_compile_options(-Wall -Wextra -O2)
endif()

include_directories(include)

# Core library
set(MINITORCH_SRCS
    src/tensor.c
    src/nn.c
    src/optim.c
    src/data.c
)

add_library(minitorch STATIC ${MINITORCH_SRCS})

# Math library on Unix / MinGW
find_library(MATH_LIB m)
if(MATH_LIB)
    target_link_libraries(minitorch PUBLIC ${MATH_LIB})
endif()

# Tests
add_executable(test_autograd tests/test_autograd.c)
target_link_libraries(test_autograd PRIVATE minitorch)

add_executable(test_nn tests/test_nn.c)
target_link_libraries(test_nn PRIVATE minitorch)

enable_testing()
add_test(NAME autograd_checks COMMAND test_autograd)
add_test(NAME module_checks COMMAND test_nn)

# Demos
add_executable(demo_xor demos/demo_xor.c)
target_link_libraries(demo_xor PRIVATE minitorch)

add_executable(demo_cnn demos/demo_cnn.c)
target_link_libraries(demo_cnn PRIVATE minitorch)

add_executable(demo_transformer demos/demo_transformer.c)
target_link_libraries(demo_transformer PRIVATE minitorch)
