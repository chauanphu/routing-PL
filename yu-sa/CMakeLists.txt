cmake_minimum_required(VERSION 3.10)
project(vrppl_loader)

set(CMAKE_CXX_STANDARD 17)

# Gather source files from new structure
file(GLOB CORE_SRC src/core/*.cpp)
file(GLOB SOLVER_SRC src/solvers/*.cpp)

add_executable(main
    src/main.cpp
    ${CORE_SRC}
    ${SOLVER_SRC}
)

add_executable(doe
    src/doe.cpp
    ${CORE_SRC}
    ${SOLVER_SRC}
)

add_executable(test
    src/test.cpp
    ${CORE_SRC}
    ${SOLVER_SRC}
)

add_executable(tune
    src/tune.cpp
    ${CORE_SRC}
    ${SOLVER_SRC}
)

find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)

target_link_libraries(main yaml-cpp OpenMP::OpenMP_CXX)
target_link_libraries(doe yaml-cpp OpenMP::OpenMP_CXX)
target_link_libraries(test yaml-cpp OpenMP::OpenMP_CXX)
target_link_libraries(tune yaml-cpp OpenMP::OpenMP_CXX)
