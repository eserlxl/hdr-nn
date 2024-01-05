#include <neuralNetwork.h>

void neuralNetwork::train(const MNISTData &trainingData, size_t miniBatchSize, float learningRate) {
    // Randomize the order of the training data to create mini-batches
    if (m_trainingOrder.size() != trainingData.NumImages()) {
        m_trainingOrder.resize(trainingData.NumImages());
        size_t index = 0;
        for (size_t &v : m_trainingOrder) {
            v = index;
            ++index;
        }
    }
#ifndef NO_RANDOMIZATION
    e2.seed(rd());
#endif
    std::shuffle(m_trainingOrder.begin(), m_trainingOrder.end(), e2);
    // Process all minibatches until we are out of training examples
    size_t trainingIndex = 0;
    while (trainingIndex < trainingData.NumImages()) {
        // Clear out minibatch derivatives. Sum up and then divide them at the end of the minibatch
        std::fill(m_miniBatchHiddenLayerBiasesDeltaCost.begin(), m_miniBatchHiddenLayerBiasesDeltaCost.end(), 0.0f);
        std::fill(m_miniBatchOutputLayerBiasesDeltaCost.begin(), m_miniBatchOutputLayerBiasesDeltaCost.end(), 0.0f);
        std::fill(m_miniBatchHiddenLayerWeightsDeltaCost.begin(), m_miniBatchHiddenLayerWeightsDeltaCost.end(), 0.0f);
        std::fill(m_miniBatchOutputLayerWeightsDeltaCost.begin(), m_miniBatchOutputLayerWeightsDeltaCost.end(), 0.0f);

        // Process the minibatch
        size_t miniBatchIndex = 0;
        while (miniBatchIndex < miniBatchSize && trainingIndex < trainingData.NumImages()) {
            // Get the training item
            uint8_t imageLabel = 0;
            const float *pixels = trainingData.GetImage(m_trainingOrder[trainingIndex], imageLabel);

            // Run the forward pass of the network
            feedForward(pixels);

            // Run the backward pass to get derivatives of the cost function
            backPropagation(pixels, imageLabel);

            /* Adding current derivatives into the minibatch derivative arrays
            we can average them at the end of the minibatch via division */
            for (size_t i = 0; i < m_hiddenLayerBiasesDeltaCost.size(); ++i)
                m_miniBatchHiddenLayerBiasesDeltaCost[i] += m_hiddenLayerBiasesDeltaCost[i];
            for (size_t i = 0; i < m_outputLayerBiasesDeltaCost.size(); ++i)
                m_miniBatchOutputLayerBiasesDeltaCost[i] += m_outputLayerBiasesDeltaCost[i];
            for (size_t i = 0; i < m_hiddenLayerWeightsDeltaCost.size(); ++i)
                m_miniBatchHiddenLayerWeightsDeltaCost[i] += m_hiddenLayerWeightsDeltaCost[i];
            for (size_t i = 0; i < m_outputLayerWeightsDeltaCost.size(); ++i)
                m_miniBatchOutputLayerWeightsDeltaCost[i] += m_outputLayerWeightsDeltaCost[i];

            // Add another item to the minibatch and used another training example
            ++trainingIndex;
            ++miniBatchIndex;
        }

        /* Divide the derivatives of the mini-series by the number of elements in
        the mini-series to get the average value of the derivatives */
        float miniBatchLearningRate = learningRate / float(miniBatchIndex);

        /* Important: Instead of doing this explicitly like in the commented code below,
        I did that implicitly above by dividing the learning rate by miniBatchIndex

        for (float& f : m_miniBatchHiddenLayerBiasesDeltaCost)  f /= float(miniBatchIndex);
        for (float& f : m_miniBatchOutputLayerBiasesDeltaCost)  f /= float(miniBatchIndex);
        for (float& f : m_miniBatchHiddenLayerWeightsDeltaCost) f /= float(miniBatchIndex);
        for (float& f : m_miniBatchOutputLayerWeightsDeltaCost) f /= float(miniBatchIndex); */

        // Application training to biases and weights
        for (size_t i = 0; i < m_hiddenLayerBiases.size(); ++i)
            m_hiddenLayerBiases[i] -= m_miniBatchHiddenLayerBiasesDeltaCost[i] * miniBatchLearningRate;
        for (size_t i = 0; i < m_outputLayerBiases.size(); ++i)
            m_outputLayerBiases[i] -= m_miniBatchOutputLayerBiasesDeltaCost[i] * miniBatchLearningRate;
        for (size_t i = 0; i < m_hiddenLayerWeights.size(); ++i)
            m_hiddenLayerWeights[i] -= m_miniBatchHiddenLayerWeightsDeltaCost[i] * miniBatchLearningRate;
        for (size_t i = 0; i < m_outputLayerWeights.size(); ++i)
            m_outputLayerWeights[i] -= m_miniBatchOutputLayerWeightsDeltaCost[i] * miniBatchLearningRate;
    }
}