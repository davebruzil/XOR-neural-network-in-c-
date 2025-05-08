#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>

using namespace std;

double relu(double x) {
    return max(0.0, x);
}

double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}
sigmoid(double x) {
    return 1 / (1 + exp(-x));
}
sigmoidDerivative(double x) {
    return x * (1 - x);
}

double meanSquaredError(const vector<double>& predictions, const vector<double>& targets) {
    double sum = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double diff = predictions[i] - targets[i];
        sum += diff * diff;
    }
    return sum / predictions.size();
}

class NeuralNetwork {
private:
    int input_size, hidden_size, output_size;
    vector<vector<double>> weights1, weights2;
    vector<double> biases1, biases2;

public:
    NeuralNetwork(int input_size, int hidden_size, int output_size)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
        weights1.resize(input_size, vector<double>(hidden_size));
        weights2.resize(hidden_size, vector<double>(output_size));
        biases1.resize(hidden_size);
        biases2.resize(output_size);
        initWeights();
    }

    void initWeights() {
        srand(time(0));
        for (auto& row : weights1)
            for (double& w : row)
                w = (double)rand() / RAND_MAX-0.5; // Random between -1 and 1

        for (auto& row : weights2)
            for (double& w : row)
                w = (double)rand()/RAND_MAX-0.5;

        for (double& b : biases1)
            b = (double)rand()/RAND_MAX-0.5;

        for (double& b : biases2)
            b = (double)rand()/RAND_MAX-0.5;
    }

    vector<double> feedforward(const vector<double> input, vector<double>& hiddenLayer) {
        hiddenLayer.resize(weights1[0].size());
        for (size_t i = 0; i < weights1.size(); ++i) {
            for (size_t j = 0; j < weights1[0].size(); ++j) {
                hiddenLayer[j] += input[i] * weights1[i][j];
            }
        }

        for (size_t j = 0; j < hiddenLayer.size(); ++j) {
            hiddenLayer[j] = sigmoid(hiddenLayer[j] + biases1[j]);
        }

        vector<double> output(weights2[0].size());
        for (size_t j = 0; j < weights2.size(); ++j) {
            for (size_t k = 0; k < weights2[0].size(); ++k) {
                output[k] += hiddenLayer[j] * weights2[j][k];
            }
        }

        for (size_t k = 0; k < output.size(); ++k) {
            output[k] = sigmoid(output[k] + biases2[k]);
        }

        return output;
    }

    vector<double> predict(const vector<double> input) {
        vector<double> hiddenLayer;
        return feedforward(input, hiddenLayer);
    }

    void train(const vector<vector<double>>& inputs,
        const vector<vector<double>>& targets,
        double learning_rate, int epochs) {
    // Input validation
    if (inputs.empty() || targets.empty() || inputs.size() != targets.size()) {
        cerr << "Invalid training data dimensions" << endl;
        return;
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            // Forward pass
            vector<double> hiddenLayer;
            vector<double> output = feedforward(inputs[i], hiddenLayer);
            
            // Calculate error
            totalError += meanSquaredError(output, targets[i]);

            // Backpropagation
            // 1. Calculate output layer gradients
            vector<double> outputGradients(output.size());
            for (size_t j = 0; j < output.size(); ++j) {
                outputGradients[j] = (output[j] - targets[i][j]) * sigmoidDerivative(output[j]);
            }

            // 2. Calculate hidden layer gradients
            vector<double> hiddenGradients(hiddenLayer.size());
            for (size_t j = 0; j < hiddenLayer.size(); ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < output.size(); ++k) {
                    sum += outputGradients[k] * weights2[j][k]; // Fixed index order
                }
                hiddenGradients[j] = sum * relu_derivative(hiddenLayer[j]);
            }

            // 3. Update output layer weights and biases
            for (size_t j = 0; j < hiddenLayer.size(); ++j) {
                for (size_t k = 0; k < output.size(); ++k) {
                    weights2[j][k] -= learning_rate * outputGradients[k] * hiddenLayer[j];
                }
            }

            for (size_t j = 0; j < output.size(); ++j) {
                biases2[j] -= learning_rate * outputGradients[j];
            }

            // 4. Update hidden layer weights and biases
            for (size_t j = 0; j < inputs[i].size(); ++j) {
                for (size_t k = 0; k < hiddenLayer.size(); ++k) {
                    weights1[j][k] -= learning_rate * hiddenGradients[k] * inputs[i][j];
                }
            }

            for (size_t j = 0; j < hiddenLayer.size(); ++j) {
                biases1[j] -= learning_rate * hiddenGradients[j];
            }
        }

        if (epoch % 1000 == 0) {
            cout << "Epoch " << (epoch + 1) << ", MSE: " << (totalError / inputs.size()) << endl;
        }
    }
}



};

int main() {
    NeuralNetwork nn(2, 4, 1);

    vector<vector<double>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    vector<vector<double>> targets = {
        {0},
        {1},
        {1},
        {0}
    };

    nn.train(inputs, targets, 0.1, 10000);

    cout << "\nPredictions after training:\n";
    for (const auto& input : inputs) {
        vector<double> output = nn.predict(input);
        cout << input[0] << " XOR " << input[1] << " = " << output[0] << endl;
    }

    return 0;
}
