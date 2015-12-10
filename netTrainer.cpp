// ECE 469: Artificial Intelligence
// Miraj Patel
// netTrainer.cpp

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "neuralNet.h"

using namespace std;

int main() {
	int numEpochs = 0;
	double learningRate = 0;
	string input, initNetwork, trainingSet, output;
	ifstream initFile, trainingFile;
	ofstream outputFile;
	
	cout << "----- Neural Network Training Program -----" << endl;
	cout << "Enter the name of the text file representing the initial neural network:" << endl;
	cin >> initNetwork;
	cout << "Enter the name of the text file representing the training set:" << endl;
	cin >> trainingSet;
	cout << "Enter a name for the output file:" << endl;
	cin >> output;
	cout << "Enter a positive integer for # of epochs:" << endl;
	cin >> input;
	numEpochs = atoi(input.c_str());
	while (numEpochs <= 0) {
		cout << "Please enter a positive integer for the # of epochs:" << endl;
		getline(cin, input);
		numEpochs = atoi(input.c_str());
	}
	cout << "Enter a number for the learning rate:" << endl;
	cin >> input;
	learningRate = atof(input.c_str());
	while (learningRate == 0) {
		cout << "Please enter a number for the learning rate:" << endl;
		getline(cin, input);
		learningRate = atof(input.c_str());
	}
	
	// Feed in initializer file into the neural network to set up the layers
	initFile.open(initNetwork.c_str(), ifstream::in);
	NeuralNetwork *myNetwork = new NeuralNetwork(initFile, numEpochs, learningRate);
	initFile.close();
	
	trainingFile.open(trainingSet.c_str(), ifstream::in);
	myNetwork->train(trainingFile);
	trainingFile.close();
	
	outputFile.open(output.c_str(), ofstream::out);
	myNetwork->saveWeights(outputFile);
	outputFile.close();
	return 0;
}