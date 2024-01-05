#include <neuralNetwork.h>

// This function evaluates the network for the given input pixels and returns the predicted label, which can range from 0 to 9
void neuralNetwork::feedForward(const float *pixels) {
    for (size_t j = 0; j < hidden_neurons; ++j) {
        float Z = m_hiddenLayerBiases[j];

        for (size_t i = 0; i < inputs; ++i)
            Z += pixels[i] * m_hiddenLayerWeights[HiddenLayerWeightIndex(i, j)];

        m_hiddenLayerOutputs[j] = 1.0f / (1.0f + std::exp(-Z));
    }

    for (size_t j = 0; j < output_neurons; ++j) {
        float Z = m_outputLayerBiases[j];

        for (size_t i = 0; i < hidden_neurons; ++i)
            Z += m_hiddenLayerOutputs[i] *
                 m_outputLayerWeights[OutputLayerWeightIndex(i, j)];

        m_outputLayerOutputs[j] = 1.0f / (1.0f + std::exp(-Z));
    }
}