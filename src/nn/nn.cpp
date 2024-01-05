#include <neuralNetwork.h>

// Write out the final weights and biases as JSON format for use in the web demo
void neuralNetwork::save() {
    FILE *file = fopen("WeightsBiases.json", "w+t");
    fprintf(file, "{\n");

    // network structure
    fprintf(file, "  \"InputNeurons\":%zu,\n", inputs);
    fprintf(file, "  \"HiddenNeurons\":%zu,\n", hidden_neurons);
    fprintf(file, "  \"OutputNeurons\":%zu,\n", output_neurons);

    // HiddenBiases
    auto hiddenBiases = GetHiddenLayerBiases();
    fprintf(file, "  \"HiddenBiases\" : [\n");
    for (size_t i = 0; i < hiddenBiases.size(); ++i) {
        fprintf(file, "    %f", hiddenBiases[i]);
        if (i < hiddenBiases.size() - 1)
            fprintf(file, ",");
        fprintf(file, "\n");
    }
    fprintf(file, "  ],\n");

    // HiddenWeights
    auto hiddenWeights = GetHiddenLayerWeights();
    fprintf(file, "  \"HiddenWeights\" : [\n");
    for (size_t i = 0; i < hiddenWeights.size(); ++i) {
        fprintf(file, "    %f", hiddenWeights[i]);
        if (i < hiddenWeights.size() - 1)
            fprintf(file, ",");
        fprintf(file, "\n");
    }
    fprintf(file, "  ],\n");

    // OutputBiases
    auto outputBiases = GetOutputLayerBiases();
    fprintf(file, "  \"OutputBiases\" : [\n");
    for (size_t i = 0; i < outputBiases.size(); ++i) {
        fprintf(file, "    %f", outputBiases[i]);
        if (i < outputBiases.size() - 1)
            fprintf(file, ",");
        fprintf(file, "\n");
    }
    fprintf(file, "  ],\n");

    // OutputWeights
    auto outputWeights = GetOutputLayerWeights();
    fprintf(file, "  \"OutputWeights\" : [\n");
    for (size_t i = 0; i < outputWeights.size(); ++i) {
        fprintf(file, "    %f", outputWeights[i]);
        if (i < outputWeights.size() - 1)
            fprintf(file, ",");
        fprintf(file, "\n");
    }
    fprintf(file, "  ]\n");

    // The end of training the neural network
    fprintf(file, "}\n");
    fclose(file);
}