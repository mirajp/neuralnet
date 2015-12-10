// ECE 469: Artificial Intelligence
// Miraj Patel
// netTester.cpp

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "neuralNet.h"

int main() {
	int numEpochs = 0;
	double learningRate = 0;
	string input, trainedNetwork, testSet, output;
	ifstream initFile, testFile;
	ofstream outputFile;
	
	cout << "----- Neural Network Testing Program -----" << endl;
	cout << "Enter the name of the text file representing the (trained) neural network:" << endl;
	cin >> trainedNetwork;
	cout << "Enter the name of the text file representing the test set:" << endl;
	cin >> testSet;
	cout << "Enter a name for the results file:" << endl;
	cin >> output;
	
	// Feed in initializer file into the neural network to set up the layers
	initFile.open(trainedNetwork.c_str(), ifstream::in);
	NeuralNetwork *myNetwork = new NeuralNetwork(initFile, numEpochs, learningRate);
	initFile.close();
	
	testFile.open(testSet.c_str(), ifstream::in);
	myNetwork->test(testFile);
	testFile.close();
	
	outputFile.open(output.c_str(), ofstream::out);
	myNetwork->saveWeights(outputFile);
	outputFile.close();
	return 0;
}