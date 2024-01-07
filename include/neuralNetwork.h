#ifndef HDR_NN_H
#define HDR_NN_H

#include <iostream>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <array>
#include <vector>
#include <algorithm>
#include "logic.h"
#include "dataLoader.h"

#define NO_RANDOMIZATION // Only for testing the algorithm, we need the same results for each run to compare.

class neuralNetwork {
public:
    std::random_device rd;
    std::mt19937 e2;

    size_t inputs;
    size_t hidden_neurons;
    size_t output_neurons;

    neuralNetwork(size_t inputs, size_t hidden_neurons, size_t output_neurons) {
        this->inputs = inputs;
        this->hidden_neurons = hidden_neurons;
        this->output_neurons = output_neurons;
        /* Set the initial weights and biases to random numbers drawn from a
        Gaussian distribution with a mean of 0 and standard deviation of 1.0 */
#ifdef NO_RANDOMIZATION
        e2.seed(1);
#else
        e2.seed(rd());
#endif
        std::normal_distribution<float> dist(0, 1);

        for (size_t i = 0; i < inputs * hidden_neurons; i++) {
            m_hiddenLayerWeights.push_back(dist(e2));
            m_hiddenLayerWeightsDeltaCost.push_back(0);
            m_miniBatchHiddenLayerWeightsDeltaCost.push_back(0);
        }
        for (size_t i = 0; i < hidden_neurons * output_neurons; i++) {
            m_outputLayerWeights.push_back(dist(e2));
            m_outputLayerWeightsDeltaCost.push_back(0);
            m_miniBatchOutputLayerWeightsDeltaCost.push_back(0);
        }
        for (size_t i = 0; i < hidden_neurons; i++) {
            m_hiddenLayerBiases.push_back(dist(e2));
            m_hiddenLayerOutputs.push_back(0);
            m_hiddenLayerBiasesDeltaCost.push_back(0);
            m_miniBatchHiddenLayerBiasesDeltaCost.push_back(0);
        }
        for (size_t i = 0; i < output_neurons; i++) {
            m_outputLayerBiases.push_back(dist(e2));
            m_outputLayerOutputs.push_back(0);
            m_outputLayerBiasesDeltaCost.push_back(0);
            m_miniBatchOutputLayerBiasesDeltaCost.push_back(0);
        }
    }

    void train(const MNISTData &trainingData, size_t miniBatchSize, float learningRate);

    void feedForward(const float *pixels);

    void backPropagation(const float *pixels, uint8_t correctLabel);

    float getDataAccuracy(const MNISTData &data) {
        size_t correctItems = 0;
        for (size_t i = 0, c = data.getImageCount(); i < c; ++i) {
            uint8_t label;
            const float *pixels = data.getImage(i, label);
            feedForward(pixels);

            // Finding the maximum value of the output layer and return the index as the label
            float maxOutput = m_outputLayerOutputs[0];
            size_t detectedLabel = 0;
            for (size_t j = 1; j < output_neurons; ++j) {
                if (m_outputLayerOutputs[j] > maxOutput) {
                    maxOutput = m_outputLayerOutputs[j];
                    detectedLabel = j;
                }
            }
            // Checking output layers for false positives, define safe range as 15%
            size_t falsePositive = 0;
            for (size_t j = 0; j < output_neurons; j++) {
                if(j==detectedLabel)
                {
                    continue;
                }
                if((maxOutput-m_outputLayerOutputs[j])/maxOutput<0.15)
                {
                    falsePositive++;
                }
            }

            if (falsePositive == 0 && detectedLabel == label)
                ++correctItems;
        }
        return float(correctItems) / float(data.getImageCount());
    }

    // Functions to get weights / bias values. They are used to make the JSON file
    const std::vector<float> &GetHiddenLayerBiases() const { return m_hiddenLayerBiases; }

    const std::vector<float> &GetOutputLayerBiases() const { return m_outputLayerBiases; }

    const std::vector<float> &GetHiddenLayerWeights() const { return m_hiddenLayerWeights; }

    const std::vector<float> &GetOutputLayerWeights() const { return m_outputLayerWeights; }

    void save();

private:

    size_t HiddenLayerWeightIndex(size_t inputIndex, size_t hiddenLayerNeuronIndex) const {
        return hiddenLayerNeuronIndex * inputs + inputIndex;
    }

    size_t OutputLayerWeightIndex(size_t hiddenLayerNeuronIndex, size_t outputLayerNeuronIndex) const {
        return outputLayerNeuronIndex * hidden_neurons + hiddenLayerNeuronIndex;
    }

private:

    // Weights and Biases
    std::vector<float> m_hiddenLayerWeights;
    std::vector<float> m_outputLayerWeights;

    std::vector<float> m_hiddenLayerBiases;
    std::vector<float> m_outputLayerBiases;

    // Neuron activation values (known as "O" values)
    std::vector<float> m_hiddenLayerOutputs;
    std::vector<float> m_outputLayerOutputs;

    // Derivatives of biases and weights for a training example
    std::vector<float> m_hiddenLayerBiasesDeltaCost;
    std::vector<float> m_outputLayerBiasesDeltaCost;

    std::vector<float> m_hiddenLayerWeightsDeltaCost;
    std::vector<float> m_outputLayerWeightsDeltaCost;

    // Average of all items in minibatch (Derivatives of biases and weights for the minibatch)
    std::vector<float> m_miniBatchHiddenLayerBiasesDeltaCost;
    std::vector<float> m_miniBatchOutputLayerBiasesDeltaCost;

    std::vector<float> m_miniBatchHiddenLayerWeightsDeltaCost;
    std::vector<float> m_miniBatchOutputLayerWeightsDeltaCost;

    // Used for minibatch generation
    std::vector<size_t> m_trainingOrder;
};

#endif // HDR_NN_H
