#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork (int inputSize_, int hiddenSize_, int outputSize_, double learningRate_) {
	inputSize = inputSize_;
	hiddenSize = hiddenSize_;
	outputSize = outputSize_;
	learningRate = learningRate_;
	weights_input_hidden.resize(hiddenSize, std::vector<double>(inputSize));
	for (int i = 0; i < hiddenSize; i++) {
		for (int j = 0; j < inputSize; j++)
		{
			weights_input_hidden[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
		}
	}
	weights_hidden_output.resize(outputSize, std::vector<double>(hiddenSize));
	for (int i = 0; i < outputSize; i++) {
		for (int j = 0; j < hiddenSize; j++) {
			weights_hidden_output[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
		}
	}
	bias_hidden.resize(hiddenSize);
	for (int i = 0; i < hiddenSize; ++i)
		bias_hidden[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

	bias_output.resize(outputSize);
	for (int i = 0; i < outputSize; ++i)
		bias_output[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

	hidden_layer_output.resize(hiddenSize);
	layer_output.resize(outputSize);

	delta_hidden.resize(hiddenSize);
	delta_output.resize(outputSize);

}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input)
{	
	for (int i = 0; i < hiddenSize; i++) {
		double sum = 0.0;
		for (int j = 0; j < inputSize; j++) {
			sum += (input[j] * weights_input_hidden[i][j]);
		}
		sum += bias_hidden[i];
		hidden_layer_output[i] = sigmoid(sum);
	}

	for (int i = 0; i < outputSize; i++) {
		double sum = 0.0;
		for (int j = 0; j < inputSize; j++) {
			sum += (hidden_layer_output[j] * weights_hidden_output[i][j]);
		}
		sum += bias_output[i];
		layer_output[i] = sigmoid(sum);
	}

	return layer_output;
}

void NeuralNetwork::backward(const std::vector<double>& input, const std::vector<double>& target)
{
	//Обчислити похідну помилки для вихідного шару
	for (int i = 0; i < outputSize; ++i) {
		delta_output[i] = (layer_output[i] - target[i]) * layer_output[i] * (1 - layer_output[i]);
	}

	//Обчислити похідну помилки для прихованого шару
	for (int i = 0; i < hiddenSize; ++i) {
		double sum = 0.0;
		for (int k = 0; k < outputSize; ++k){
			sum += (delta_output[k] * weights_hidden_output[k][i]);
		}
		delta_hidden[i] = sum * hidden_layer_output[i] * (1 - hidden_layer_output[i]);
	}	

	//оновлення ваг: Від прихованого до вихідного шару
	for (int i = 0; i < outputSize; ++i) {
		for (int j = 0; j < hiddenSize; ++j) {
			weights_hidden_output[i][j] -= learningRate * delta_output[i] * hidden_layer_output[j];
		}
		bias_output[i] -= learningRate * delta_output[i];
	}

	//оновлення ваг: Від вхідного до прихованого шару
	for (int i = 0; i < hiddenSize; ++i) {
		for (int j = 0; j < inputSize; ++j) {
			weights_input_hidden[i][j] -= learningRate * delta_hidden[i] * input[j];
		}
		bias_hidden[i] -= learningRate * delta_hidden[i];  
	}
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs)
{
	for (int epoch = 0; epoch < epochs; ++epoch) {
		double total_loss = 0.0;
		for (int i = 0; i < inputs.size(); ++i) {
			std::vector<double> output = forward(inputs[i]);
			backward(inputs[i], targets[i]);
			double diff = output[0] - targets[i][0];
			total_loss += diff * diff;  
		}
		if (epoch < 20 || epoch % 1000 == 0) {
			total_loss /= inputs.size();  
			std::cout << "Epoch: " << epoch << "  Loss: " << total_loss << std::endl;
		}
	}
}

double NeuralNetwork::sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}
