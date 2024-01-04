#include <neuralNetwork.h>

// This function evaluates the network for the given input pixels and returns the predicted label, which can range from 0 to 9
void neuralNetwork::feedForward(const float *pixels) {
    for (size_t i = 0; i < hidden_neurons; ++i) {
        float Z = m_hiddenLayerBiases[i];

        for (size_t inputIndex = 0; inputIndex < inputs; ++inputIndex)
            Z += pixels[inputIndex] * m_hiddenLayerWeights[HiddenLayerWeightIndex(inputIndex, i)];

        m_hiddenLayerOutputs[i] = 1.0f / (1.0f + std::exp(-Z));
    }

    for (size_t i = 0; i < output_neurons; ++i) {
        float Z = m_outputLayerBiases[i];

        for (size_t inputIndex = 0; inputIndex < hidden_neurons; ++inputIndex)
            Z += m_hiddenLayerOutputs[inputIndex] *
                 m_outputLayerWeights[OutputLayerWeightIndex(inputIndex, i)];

        m_outputLayerOutputs[i] = 1.0f / (1.0f + std::exp(-Z));
    }
}