// ECE 469: Artificial Intelligence
// Miraj Patel
// neuralNet.cpp

#include "neuralNet.h"
#include <iomanip>
#include <cmath>

NeuralNetwork::NeuralNetwork(std::ifstream &inputFile, int numEpochs, double learningRate) {
	numEpochs = numEpochs;
	learningRate = learningRate;
	
	// This input file initializes the parameters and weights of the new neural network
	inputFile >> numInput;
	inputFile >> numHidden;
	inputFile >> numOutput;
	double weight = 0;
	//std::cout << "numInput = " << numInput << ", numHidden = " << numHidden << ", numOutput = " << numOutput << std::endl;
	
	inputActivations.resize(numInput+1, 0);
	hiddenActivations.resize(numHidden+1, 0);
	outputActivations.resize(numOutput, 0);
	inputActivations[0] = -1;
	hiddenActivations[0] = -1;
	// Output layer doesn't have bias weight
	
	hiddenDeltas.resize(numHidden+1, 0);
	outputDeltas.resize(numOutput, 0);
	
	// numHidden lines with numInput+1 weights on each line
	for (int hiddenIter = 0; hiddenIter < numHidden; hiddenIter++) {
		std::vector <double> weights;
		for (int weightIter = 0; weightIter <= numInput; weightIter++) {
			inputFile >> weight;
			weights.push_back(weight);
		}
		// Push the whole row of weights of edges that go into hidden node #iter+1
		hiddenWeights.push_back(weights);
	}
	
	for (int outputIter = 0; outputIter < numOutput; outputIter++) {
		std::vector <double> weights;
		for (int weightIter = 0; weightIter <= numHidden; weightIter++) {
			inputFile >> weight;
			weights.push_back(weight);
		}
		// Push the whole row of weights of edges that go into output node #iter+1
		outputWeights.push_back(weights);
	}
}

void NeuralNetwork::train(std::ifstream &inputFile) {
	// Back-Prop-Learning pseudocode implementation
	// This assumes the network's weights have been initialized, and a training file is fed in to train the weights
	int numExamples;
	inputFile >> numExamples;
	for (int epoch = 0; epoch < numEpochs; epoch++) {
		// Get a single example: vector of inputs and outputs
		std::vector <double> exampleXs;
		std::vector <int> exampleYs;
		double exX;
		int exY;
		for (int inputIter = 0; inputIter < numInput; inputIter++) {
			inputFile >> exX;
			// To compute activations for layer 1 (input):
			// Copy input vector of a single example to the input nodes of the network
			inputActivations[inputIter+1] = exX;
			exampleXs.push_back(exX);
		}
		for (int outputIter = 0; outputIter < numOutput; outputIter++) {
			inputFile >> exY;
			exampleYs.push_back(exY);
		}
		
		// Propogate the inputs forward to compute outputs
		// To compute activations for layer 2 (hidden):
		for (int hiddenIter = 0; hiddenIter < numHidden; hiddenIter++) {
			// Compute sum of weight of all input nodes to this node in the hidden layer * activation in the input layer
			double sum = 0;
			// total number of terms = number of input nodes + bias 
			for (int inputIter = 0; inputIter <= numInput+1; inputIter++) {
				sum += ((hiddenWeights[hiddenIter][inputIter])*inputActivations[inputIter]);
			}
			// Use the sum as input to jth node, and apply the activation function (sigmoid)
			// Skip index 0 since it corresponds to -1 activation for the bias weight
			hiddenActivations[hiddenIter+1] = sigmoid(sum);
		}
		// To compute activations for layer 3 (output):
		for (int outputIter = 0; outputIter < numOutput; outputIter++) {
			double sum = 0;
			for (int hiddenIter = 0; hiddenIter <= numHidden+1; hiddenIter++) {
				sum += ((outputWeights[outputIter][hiddenIter])*hiddenActivations[hiddenIter]);
			}
			outputActivations[outputIter] = sigmoid(sum);
		}
		
		// Propogate deltas backward from output layer to input layer
		//outputDeltas.clear();
		//hiddenDeltas.clear();
		for (int outputIter = 0; outputIter < numOutput; outputIter++) {
			outputDeltas[outputIter] = derivSigmoid(outputActivations[outputIter])*(exampleYs[outputIter] - outputActivations[outputIter]);
		}
		
		// Propogate deltas for the hidden layer
		for (int hiddenIter = 0; hiddenIter <= numHidden; hiddenIter++) {
			double sum = 0;
			for (int outputIter = 0; outputIter < numOutput; outputIter++) {
				sum += (outputWeights[outputIter][hiddenIter] * outputDeltas[outputIter]);
			}
			hiddenDeltas[hiddenIter] = derivSigmoid(hiddenActivations[hiddenIter])*sum;
		}
		
		// Update weights
		for (int inputIter = 0; inputIter <= numInput; inputIter++) {
			for (int hiddenIter = 0; hiddenIter < numHidden; hiddenIter++) {
				hiddenWeights[hiddenIter][inputIter] = hiddenWeights[hiddenIter][inputIter] + (learningRate*inputActivations[inputIter]*hiddenDeltas[hiddenIter]);
			}
		}
		for (int hiddenIter = 0; hiddenIter <= numHidden; hiddenIter++) {
			for (int outputIter = 0; outputIter < numOutput; outputIter++) {
				outputWeights[outputIter][hiddenIter] = outputWeights[outputIter][hiddenIter] + (learningRate*hiddenActivations[hiddenIter]*outputDeltas[outputIter]);
			}
		}
	}
	// After # of epochs of training completed, return
	return;
}

void NeuralNetwork::saveWeights(std::ofstream &outputFile) {
	outputFile << numInput << " " << numHidden << " " << numOutput << std::endl;
	for (int hiddenIter = 0; hiddenIter < numHidden; hiddenIter++) {
		for (int inputIter = 0; inputIter < numInput; inputIter++) {	
			outputFile << std::fixed << std::setprecision(3) << hiddenWeights[hiddenIter][inputIter] << " ";
		}
		outputFile << std::fixed << std::setprecision(3) << hiddenWeights[hiddenIter][numInput] << std::endl;
	}
	for (int outputIter = 0; outputIter < numOutput; outputIter++) {
		for (int hiddenIter = 0; hiddenIter < numHidden; hiddenIter++) {	
			outputFile << std::fixed << std::setprecision(3) << hiddenWeights[outputIter][hiddenIter] << " ";
		}
		outputFile << std::fixed << std::setprecision(3) << hiddenWeights[outputIter][numHidden] << std::endl;
	}
}

double NeuralNetwork::sigmoid(double inputVal) {
	return (1/(1+exp(-inputVal)));
}

double NeuralNetwork::derivSigmoid(double inputVal) {
	return (sigmoid(inputVal)*(1-sigmoid(inputVal)));
}

