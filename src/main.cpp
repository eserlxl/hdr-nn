#include "timer.h"
#include "dataLoader.h"
#include "neuralNetwork.h"

// Setting to "1" shows error after each training and writes them in Error.csv file
// It may slow down the training process by 50%.
#define REPORT_ERROR_WHILE_TRAINING() 0

const size_t c_trainingEpochs = 10;
const size_t c_miniBatchSize = 10;
const float c_learningRate = 10.f;

// The datasets used for training and testing a model
MNISTData g_trainingData;
MNISTData g_testData;

int main(int argc, char **argv) {

    // Loading the MNIST data
    if (!g_trainingData.load(true) || !g_testData.load(false)) {
        printf("Could not load the MNIST data!\n");
        return 1;
    }

    auto *nn = new neuralNetwork(785, 1000, 10);

    //printf("\nInitial training data accuracy: %0.3f%%\n", 100.0f * nn->getDataAccuracy(g_trainingData));
    //printf("\nInitial test data accuracy: %0.3f%%\n\n", 100.0f * nn->getDataAccuracy(g_testData));

#if REPORT_ERROR_WHILE_TRAINING()
    FILE *file = fopen("Error.csv","w+t");
    if (!file)
    {
        printf("Could not open 'Error.csv' for writing!\n");
        return 2;
    }
    fprintf(file, "\"Training Data Accuracy\",\"Testing Data Accuracy\"\n");
#endif

    {
        Timer timer("The training time:  ");

        // We report error before each training of neural network
        for (size_t epoch = 0; epoch < c_trainingEpochs; ++epoch) {
#if REPORT_ERROR_WHILE_TRAINING()
            float accuracyTraining = GetDataAccuracy(g_trainingData);
            float accuracyTest = GetDataAccuracy(g_testData);
            printf("Training data accuracy: %0.2f%%\n", 100.0f*accuracyTraining);
            printf("Test data accuracy: %0.2f%%\n\n", 100.0f*accuracyTest);
            fprintf(file, "\"%f\",\"%f\"\n", accuracyTraining, accuracyTest);
#endif

            printf("Training the epoch %zu / %zu...\n", epoch + 1, c_trainingEpochs);
            nn->train(g_trainingData, c_miniBatchSize, c_learningRate);
            printf("\n");
        }
    }

    // report final error
    float accuracyTraining = nn->getDataAccuracy(g_trainingData);
    float accuracyTest = nn->getDataAccuracy(g_testData);
    printf("\nTraining/Test Accuracy: %0.3f%% / %0.3f%%\n", 100.0f * accuracyTraining, 100.0f * accuracyTest);

#if REPORT_ERROR_WHILE_TRAINING()
    fprintf(file, "\"%f\",\"%f\"\n", accuracyTraining, accuracyTest);
    fclose(file);
#endif

    nn->save();
    delete (nn);

    return 0;
}
