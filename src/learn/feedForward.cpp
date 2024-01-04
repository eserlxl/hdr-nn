#include <neuralNetwork.h>

// This function evaluates the network for the given input pixels and returns the predicted label, which can range from 0 to 9
uint8_t neuralNetwork::feedForward(const float *pixels, uint8_t correctLabel) {
    for (size_t neuronIndex = 0; neuronIndex < hidden_neurons; ++neuronIndex) {
        float Z = m_hiddenLayerBiases[neuronIndex];

        for (size_t inputIndex = 0; inputIndex < inputs; ++inputIndex)
            Z += pixels[inputIndex] * m_hiddenLayerWeights[HiddenLayerWeightIndex(inputIndex, neuronIndex)];

        m_hiddenLayerOutputs[neuronIndex] = 1.0f / (1.0f + std::exp(-Z));
    }

    for (size_t neuronIndex = 0; neuronIndex < output_neurons; ++neuronIndex) {
        float Z = m_outputLayerBiases[neuronIndex];

        for (size_t inputIndex = 0; inputIndex < hidden_neurons; ++inputIndex)
            Z += m_hiddenLayerOutputs[inputIndex] *
                 m_outputLayerWeights[OutputLayerWeightIndex(inputIndex, neuronIndex)];

        m_outputLayerOutputs[neuronIndex] = 1.0f / (1.0f + std::exp(-Z));
    }

    // Finding the maximum value of the output layer and return the index as the label
    float maxOutput = m_outputLayerOutputs[0];
    uint8_t maxLabel = 0;
    for (uint8_t neuronIndex = 1; neuronIndex < output_neurons; ++neuronIndex) {
        if (m_outputLayerOutputs[neuronIndex] > maxOutput) {
            maxOutput = m_outputLayerOutputs[neuronIndex];
            maxLabel = neuronIndex;
        }
    }
    return maxLabel;
}