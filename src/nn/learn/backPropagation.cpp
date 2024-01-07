#include <neuralNetwork.h>

/* This function calculates the gradient needed for training by backpropagating the error of
the network, using the neuron output values from the forward pass. It determines the error
by comparing the label predicted by the network to the correct label */

void neuralNetwork::backPropagation(const float *pixels, uint8_t correctLabel) {
    // Since we are proceeding backwards, we are starting with the output layer
    for (size_t j = 0; j < output_neurons; j++) {
        float desiredOutput = (correctLabel == j) ? 1.0f : 0.0f;

        float deltaCost_deltaO = m_outputLayerOutputs[j] - desiredOutput;
        float deltaO_deltaZ = m_outputLayerOutputs[j] * (1.0f - m_outputLayerOutputs[j]);

        m_outputLayerBiasesDeltaCost[j] = deltaCost_deltaO * deltaO_deltaZ;

        // Calculating deltaCost/deltaWeight for each weight going into the neuron
        for (size_t i = 0; i < hidden_neurons; i++) {
            m_outputLayerWeightsDeltaCost[OutputLayerWeightIndex(i, j)] =
                    m_outputLayerBiasesDeltaCost[j] * m_hiddenLayerOutputs[i];
        }
    }

    for (size_t j = 0; j < hidden_neurons; j++) {
        /* To calculate the error (deltaCost/deltaBias) for each hidden neuron we are following these steps:

        1. Multiply the deltaCost/deltaDestinationZ, which is already calculated and stored in
        m_outputLayerBiasesDeltaCost[destinationNeuronIndex], by the weight connecting the source and target neurons.
        This multiplication gives the error value for the neuron
        2. Multiply the neuron's output (O) by (1 - O) to obtain deltaO/deltaZ
        3. Compute deltaCost/deltaZ by multiplying the error by deltaO/deltaZ

        By following these steps, you can calculate the error (deltaCost/deltaBias) for each hidden neuron */

        float deltaCost_deltaO = 0.0f;
        for (size_t i = 0; i < output_neurons; i++) {
            deltaCost_deltaO += m_outputLayerBiasesDeltaCost[i] * m_outputLayerWeights[OutputLayerWeightIndex(j, i)];
        }

        float deltaO_deltaZ = m_hiddenLayerOutputs[j] * (1.0f - m_hiddenLayerOutputs[j]);
        m_hiddenLayerBiasesDeltaCost[j] = deltaCost_deltaO * deltaO_deltaZ;

        // Calculating deltaCost/deltaWeight for each weight going into the neuron
        for (size_t i = 0; i < inputs; i++) {
            m_hiddenLayerWeightsDeltaCost[HiddenLayerWeightIndex(i, j)] = m_hiddenLayerBiasesDeltaCost[j] * pixels[i];
        }

    }
}