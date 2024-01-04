#include <neuralNetwork.h>

/* This function calculates the gradient needed for training by backpropagating the error of
the network, using the neuron output values from the forward pass. It determines the error
by comparing the label predicted by the network to the correct label */

void neuralNetwork::backPropagation(const float *pixels, uint8_t correctLabel) {
    // Since we are proceeding backwards, we are starting with the output layer
    for (size_t neuronIndex = 0; neuronIndex < output_neurons; ++neuronIndex) {
        float desiredOutput = (correctLabel == neuronIndex) ? 1.0f : 0.0f;

        float deltaCost_deltaO = m_outputLayerOutputs[neuronIndex] - desiredOutput;
        float deltaO_deltaZ = m_outputLayerOutputs[neuronIndex] * (1.0f - m_outputLayerOutputs[neuronIndex]);

        m_outputLayerBiasesDeltaCost[neuronIndex] = deltaCost_deltaO * deltaO_deltaZ;

        // Calculating deltaCost/deltaWeight for each weight going into the neuron
        for (size_t inputIndex = 0; inputIndex < hidden_neurons; ++inputIndex)
            m_outputLayerWeightsDeltaCost[OutputLayerWeightIndex(inputIndex, neuronIndex)] =
                    m_outputLayerBiasesDeltaCost[neuronIndex] * m_hiddenLayerOutputs[inputIndex];
    }

    for (size_t neuronIndex = 0; neuronIndex < hidden_neurons; ++neuronIndex) {
        /* To calculate the error (deltaCost/deltaBias) for each hidden neuron we are following these steps:

        1. Multiply the deltaCost/deltaDestinationZ, which is already calculated and stored in
        m_outputLayerBiasesDeltaCost[destinationNeuronIndex], by the weight connecting the source and target neurons.
        This multiplication gives the error value for the neuron
        2. Multiply the neuron's output (O) by (1 - O) to obtain deltaO/deltaZ
        3. Compute deltaCost/deltaZ by multiplying the error by deltaO/deltaZ

        By following these steps, you can calculate the error (deltaCost/deltaBias) for each hidden neuron */

        float deltaCost_deltaO = 0.0f;
        for (size_t destinationNeuronIndex = 0; destinationNeuronIndex < output_neurons; ++destinationNeuronIndex)
            deltaCost_deltaO += m_outputLayerBiasesDeltaCost[destinationNeuronIndex] *
                                m_outputLayerWeights[OutputLayerWeightIndex(neuronIndex, destinationNeuronIndex)];
        float deltaO_deltaZ = m_hiddenLayerOutputs[neuronIndex] * (1.0f - m_hiddenLayerOutputs[neuronIndex]);
        m_hiddenLayerBiasesDeltaCost[neuronIndex] = deltaCost_deltaO * deltaO_deltaZ;

        // Calculating deltaCost/deltaWeight for each weight going into the neuron
        for (size_t inputIndex = 0; inputIndex < inputs; ++inputIndex)
            m_hiddenLayerWeightsDeltaCost[HiddenLayerWeightIndex(inputIndex, neuronIndex)] =
                    m_hiddenLayerBiasesDeltaCost[neuronIndex] * pixels[inputIndex];
    }
}