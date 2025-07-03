#pragma once
#include <iostream>
#include <vector>
#include <cmath>
class NeuralNetwork
{
public:

	NeuralNetwork(int inputSize_, int hiddenSize_, int outputSize_, double learningRate_);

	int inputSize, hiddenSize, outputSize;
	double learningRate;

	std::vector<std::vector<double>> weights_input_hidden;
	std::vector<std::vector<double>> weights_hidden_output;

	std::vector<double> bias_hidden;
	std::vector<double> bias_output;

	std::vector<double> hidden_layer_output;
	std::vector<double> layer_output;

	std::vector<double> delta_hidden;
	std::vector<double> delta_output;

	double sigmoid(double x); 

	std::vector<double> forward(const std::vector<double>& input);
	void backward(const std::vector<double>& input, const std::vector<double>& target);
	void train(const std::vector<std::vector<double>>& inputs,const std::vector<std::vector<double>>& targets,int epochs);
	
};

