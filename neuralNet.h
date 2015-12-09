// ECE 469: Artificial Intelligence
// Miraj Patel
// neuralNet.h

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>

class NeuralNetwork {
	public:
		// Constructor loads files and sets values and weights
		NeuralNetwork(std::ifstream &inputFile, int numEpochs, double learningRate);
		
		// Ni = number of input nodes, Nh = number of hidden nodes, No = number of output nodes
		int numEpochs, numInput, numHidden, numOutput;
		double learningRate;
		// vector of vector of double for weights of the edges from all the input nodes (and bias) to each hidden nodes
		// and weights for each edge from each hidden node (and bias) to each output node
		// size of hiddenWeights: Nh x Ni+1, size of outputWeights: No x Nh + 1
		std::vector <std::vector <double> > hiddenWeights, outputWeights;
		
		// activations of each node
		std::vector <double> inputActivations, hiddenActivations, outputActivations;
		
		// deltas: difference from level l+1 and l, thus only for hidden layer and output layer
		std::vector <double> hiddenDeltas, outputDeltas;
		
		// activation function: sigmoid
		double sigmoid(double inputVal);
		
		// derivative of activation function: derivSigmoid
		double derivSigmoid(double inputVal);
		
		void train(std::ifstream &inputFile);
		
		void saveWeights(std::ofstream &outputFile);
};


#endif