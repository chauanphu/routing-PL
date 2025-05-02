cmake_minimum_required(VERSION 3.10)
project(vrppl_loader)

set(CMAKE_CXX_STANDARD 17)

add_executable(vrppl_loader
    src/main.cpp
    src/Node.cpp
    src/Customer.cpp
    src/Locker.cpp
    src/Depot.cpp
    src/VRPInstance.cpp
    src/InstanceParser.cpp
    src/Solver.cpp
    src/SA.cpp
    src/GA.cpp
)

find_package(yaml-cpp REQUIRED)
target_link_libraries(vrppl_loader yaml-cpp)
