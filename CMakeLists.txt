cmake_minimum_required(VERSION 3.16)
project(hdr-nn)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast")

include_directories(include)

add_executable(hdr-nn
        include/neuralNetwork.h include/timer.h include/dataLoader.h src/nn/learn/train.cpp
        src/main.cpp src/nn/learn/feedForward.cpp src/nn/learn/backPropagation.cpp src/nn/nn.cpp include/logic.h)
