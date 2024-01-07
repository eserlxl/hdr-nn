#ifndef HDR_NN_LOGIC_H
#define HDR_NN_LOGIC_H

#include <cmath>

#define logic sigmoid

/**
 * Sigmoid
 * Simple logistic function, It is a smooth, S-shaped curve.
 * Input: [0,1], Output: [0,1]
 */
template<typename Float>
Float sigmoid(Float x, Float a=46.875) {
    return (Float) 1 / ((Float) 1 + std::exp(-a*x));
}

#endif //HDR_NN_LOGIC_H
