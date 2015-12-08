// ECE 469: Artificial Intelligence
// Miraj Patel
// neuralNet.cpp

#include "neuralNet.h"
#include <cmath>

NeuralNetwork::NeuralNetwork(std::ifstream &inputFile) {
	
}

double NeuralNetwork::sigmoid(double inputVal) {
	return (1/(1+exp(-inputVal)));
}

double NeuralNetwork::derivSigmoid(double inputVal) {
	return (sigmoid(inputVal)*(1-sigmoid(inputVal)));
}

