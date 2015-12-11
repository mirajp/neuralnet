// ECE 469: Artificial Intelligence
// Miraj Patel
// neuralNet.cpp

#include "neuralNet.h"
#include <iomanip>
#include <cmath>

NeuralNetwork::NeuralNetwork(std::ifstream &inputFile, int epochs, double rate) {
	this->numEpochs = epochs;
	this->learningRate = rate;
	
	// This input file initializes the parameters and weights of the new neural network
	inputFile >> numInput;
	inputFile >> numHidden;
	inputFile >> numOutput;
	double weight = 0;
	
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
	for (int epoch = 0; epoch < numEpochs; epoch++) {
		int numExamples;
		int drop;
		inputFile >> numExamples;
		inputFile >> drop;
		inputFile >> drop;
		
		for (int exIter = 0; exIter < numExamples; exIter++) {
			// reset deltas and activation levels to 0 after each example
			fill(hiddenDeltas.begin(), hiddenDeltas.end(), 0);
			fill(outputDeltas.begin(), outputDeltas.end(), 0);
			fill(inputActivations.begin(), inputActivations.end(), 0);
			fill(hiddenActivations.begin(), hiddenActivations.end(), 0);
			fill(outputActivations.begin(), outputActivations.end(), 0);
			inputActivations[0] = -1;
			hiddenActivations[0] = -1;
			
			// Get a single example: vector of inputs and outputs
			std::vector <int> exampleYs;
			double exX;
			int exY;
			// 0th index is activation value for bias weight = -1
			for (int inputIter = 1; inputIter <= numInput; inputIter++) {
				inputFile >> exX;
				// To compute activations for layer 1 (input):
				// Copy input vector of a single example to the input nodes of the network
				inputActivations[inputIter] = exX;
			}
			for (int outputIter = 0; outputIter < numOutput; outputIter++) {
				inputFile >> exY;
				exampleYs.push_back(exY);
			}
			
			// Propagate the inputs forward to compute outputs
			// To compute activations for layer 2 (hidden):
			for (int hiddenIter = 0; hiddenIter < numHidden; hiddenIter++) {
				// Compute sum of weight of all input nodes to this node in the hidden layer * activation in the input layer
				double sum = 0;
				// total number of terms = number of input nodes + bias 
				for (int inputIter = 0; inputIter <= numInput; inputIter++) {
					sum += ((hiddenWeights[hiddenIter][inputIter])*inputActivations[inputIter]);
				}
				// Use the sum as input to jth node, and apply the activation function (sigmoid)
				// Skip index 0 since it corresponds to -1 activation for the bias weight
				hiddenActivations[hiddenIter+1] = sigmoid(sum);
			}
			for (int i = 0; i < hiddenActivations.size(); i++) {
				//std::cout << "hiddenActivations[" << i << "] = " << hiddenActivations[i] << std::endl;
			}
			// To compute activations for layer 3 (output):
			for (int outputIter = 0; outputIter < numOutput; outputIter++) {
				double sum = 0;
				for (int hiddenIter = 0; hiddenIter <= numHidden; hiddenIter++) {
					sum += ((outputWeights[outputIter][hiddenIter])*hiddenActivations[hiddenIter]);
				}
				outputActivations[outputIter] = sigmoid(sum);
			}
			for (int i = 0; i < outputActivations.size(); i++) {
				//std::cout << "outputActivations[" << i << "] = " << outputActivations[i] << std::endl;
			}
			// Propagate deltas backward from output layer to input layer
			for (int outputIter = 0; outputIter < numOutput; outputIter++) {
				outputDeltas[outputIter] = (outputActivations[outputIter]*(1-outputActivations[outputIter]))*(exampleYs[outputIter] - outputActivations[outputIter]);
				//std::cout << "outputDeltas[" << outputIter << "] = " << outputDeltas[outputIter] << std::endl;
			}
			
			// Propagate deltas for the hidden layer
			//for (int hiddenIter = 1; hiddenIter <= numHidden; hiddenIter++) {
			for (int hiddenIter = 0; hiddenIter <= numHidden; hiddenIter++) { // include bias weight too
				double sum = 0;
				for (int outputIter = 0; outputIter < numOutput; outputIter++) {
					sum += (outputWeights[outputIter][hiddenIter] * outputDeltas[outputIter]);
				}
				hiddenDeltas[hiddenIter] = (hiddenActivations[hiddenIter]*(1-hiddenActivations[hiddenIter]))*sum;
				//std::cout << "hiddenDeltas[" << hiddenIter << "] = " << hiddenDeltas[hiddenIter] << std::endl;
			}
			//std::cout << "UPDATING WEIGHTS:" << std::endl;
			// Update weights
			//for (int hiddenIter = 0; hiddenIter < numHidden; hiddenIter++) {
			for (int hiddenIter = 0; hiddenIter <= numHidden; hiddenIter++) {
				for (int outputIter = 0; outputIter < numOutput; outputIter++) {
					//std::cout << "hidden node #" << hiddenIter << ", output node #" << outputIter << " = " << outputWeights[outputIter][hiddenIter] << ", changing by: " << (learningRate*hiddenActivations[hiddenIter]*outputDeltas[outputIter]) << std::endl;
					outputWeights[outputIter][hiddenIter] = outputWeights[outputIter][hiddenIter] + (learningRate*hiddenActivations[hiddenIter]*outputDeltas[outputIter]);
				}
			}
			for (int inputIter = 0; inputIter <= numInput; inputIter++) {
				for (int hiddenIter = 0; hiddenIter < numHidden; hiddenIter++) { // do not count bias node in hidden layer
					//hiddenWeights[hiddenIter][inputIter] = hiddenWeights[hiddenIter][inputIter] + (learningRate*inputActivations[inputIter]*hiddenDeltas[hiddenIter+1]);
					//std::cout << "input node #" << inputIter << ", hidden node #" << (hiddenIter+1) << " = " << hiddenWeights[hiddenIter][inputIter] << ", changing by: " << (learningRate*inputActivations[inputIter]*hiddenDeltas[hiddenIter+1]) << std::endl;
					hiddenWeights[hiddenIter][inputIter] = hiddenWeights[hiddenIter][inputIter] + (learningRate*inputActivations[inputIter]*hiddenDeltas[hiddenIter+1]);
				}
			}
		}
		// Faster to have it in memory, but time is not a constraint so will just keep reading from file
		inputFile.seekg(0, inputFile.beg);
	}
	// After # of epochs of training completed, return
	return;
}

void NeuralNetwork::test(std::ifstream &inputFile) {
	int numExamples;
	int drop;
	inputFile >> numExamples;
	inputFile >> drop;
	inputFile >> drop;
	
	aCount.clear();
	bCount.clear();
	cCount.clear();
	dCount.clear();
	globalA = 0;
	globalB = 0;
	globalC = 0;
	globalD = 0;
	
	for (int classIter = 0; classIter < numOutput; classIter++) {
		aCount.push_back(0);
		bCount.push_back(0);
		cCount.push_back(0);
		dCount.push_back(0);
		accuracy.push_back(0);
		precision.push_back(0);
		recall.push_back(0);
		f1.push_back(0);
	}
	
	for (int exIter = 0; exIter < numExamples; exIter++) {
		// reset activation levels to 0 after each example
		fill(inputActivations.begin(), inputActivations.end(), 0);
		fill(hiddenActivations.begin(), hiddenActivations.end(), 0);
		fill(outputActivations.begin(), outputActivations.end(), 0);
		inputActivations[0] = -1;
		hiddenActivations[0] = -1;
		
		// Get a single example: vector of inputs and outputs
		std::vector <int> exampleYs;
		double exX;
		int exY;
		// 0th index is activation value for bias weight = -1
		for (int inputIter = 1; inputIter <= numInput; inputIter++) {
			inputFile >> exX;
			// To compute activations for layer 1 (input):
			// Copy input vector of a single example to the input nodes of the network
			inputActivations[inputIter] = exX;
		}
		for (int outputIter = 0; outputIter < numOutput; outputIter++) {
			inputFile >> exY;
			exampleYs.push_back(exY);
		}
		
		// Propagate the inputs forward to compute outputs
		// To compute activations for layer 2 (hidden):
		for (int hiddenIter = 0; hiddenIter < numHidden; hiddenIter++) {
			// Compute sum of weight of all input nodes to this node in the hidden layer * activation in the input layer
			double sum = 0;
			// total number of terms = number of input nodes + bias 
			for (int inputIter = 0; inputIter <= numInput; inputIter++) {
				sum += ((hiddenWeights[hiddenIter][inputIter])*inputActivations[inputIter]);
			}
			// Use the sum as input to jth node, and apply the activation function (sigmoid)
			// Skip index 0 since it corresponds to -1 activation for the bias weight
			hiddenActivations[hiddenIter+1] = sigmoid(sum);
		}
		// To compute activations for layer 3 (output):
		for (int outputIter = 0; outputIter < numOutput; outputIter++) {
			double sum = 0;
			for (int hiddenIter = 0; hiddenIter <= numHidden; hiddenIter++) {
				sum += ((outputWeights[outputIter][hiddenIter])*hiddenActivations[hiddenIter]);
			}
			outputActivations[outputIter] = sigmoid(sum);
		}
		
		analyzeResult(exampleYs);
	}
}

void NeuralNetwork::analyzeResult(std::vector <int> &expectedOutput) {
	for (int classIter = 0; classIter < numOutput; classIter++) {
		double receivedOutput = outputActivations[classIter];
		
		if (receivedOutput >= 0.5) {
			// round to an output of 1
			if (expectedOutput[classIter] == 1) {
				aCount[classIter]++;
				globalA++;
				//std::cout << "globalA = " << globalA << std::endl;
			}
			else {
				bCount[classIter]++;
				globalB++;
			}
		}
		
		else {
			// round to output of 0
			if (expectedOutput[classIter] == 1) {
				cCount[classIter]++;
				globalC++;
			}
			else {
				dCount[classIter]++;
				globalD++;
				//std::cout << "globalD = " << globalD << std::endl;
			}
		}
	}
}

void NeuralNetwork::saveResults(std::ofstream &outputFile) {
	double microAcc, microPrec, microRecall, microF1, macroF1, macroAcc = 0, macroPrec = 0, macroRecall = 0;
	for (int classIter = 0; classIter < numOutput; classIter++) {
		accuracy[classIter] = (double) (aCount[classIter] + dCount[classIter])/(aCount[classIter] + bCount[classIter] + cCount[classIter] + dCount[classIter]);
		macroAcc += accuracy[classIter];
		
		precision[classIter] = (double) (aCount[classIter])/(aCount[classIter] + bCount[classIter]);
		macroPrec += precision[classIter];
		
		recall[classIter] = (double) (aCount[classIter])/(aCount[classIter] + cCount[classIter]);
		macroRecall += recall[classIter];
		
		f1[classIter] = (double) (2*precision[classIter]*recall[classIter])/(precision[classIter] + recall[classIter]);
		
		//outputFile << std::fixed << std::setprecision(0) << aCount[classIter] << " " << std::fixed << std::setprecision(0) << bCount[classIter] << " " << std::fixed << std::setprecision(0) << cCount[classIter] << " " << std::fixed << std::setprecision(0) << dCount[classIter] << " " << std::fixed << std::setprecision(3) << accuracy[classIter] << " " << std::fixed << std::setprecision(3) << precision[classIter] << " " << std::fixed << std::setprecision(3) << recall[classIter] << " " << std::fixed << std::setprecision(3) << f1[classIter] << std::endl;
		outputFile << std::fixed << std::setprecision(0) << aCount[classIter] <<
		" " << std::fixed << std::setprecision(0) << bCount[classIter] <<
		" " << std::fixed << std::setprecision(0) << cCount[classIter] <<
		" " << std::fixed << std::setprecision(0) << dCount[classIter] <<
		" " << std::fixed << std::setprecision(3) << accuracy[classIter] <<
		" " << std::fixed << std::setprecision(3) << precision[classIter] <<
		" " << std::fixed << std::setprecision(3) << recall[classIter] <<
		" " << std::fixed << std::setprecision(3) << f1[classIter] << std::endl;
	}
	
	microAcc = (double) (globalA + globalD)/(globalA + globalB + globalC + globalD);
	microPrec = (double) (globalA)/(globalA + globalB);
	microRecall = (double) (globalA)/(globalA + globalC);
	microF1 = (double) (2*microPrec*microRecall)/(microPrec + microRecall);
	
	outputFile << std::fixed << std::setprecision(3) << microAcc << " " << std::fixed << std::setprecision(3) << microPrec << " " << std::fixed << std::setprecision(3) << microRecall << " " << std::fixed << std::setprecision(3) << microF1 << std::endl;
	
	macroAcc = (double) (macroAcc/numOutput);
	macroPrec = (double) (macroPrec/numOutput);
	macroRecall = (double) (macroRecall/numOutput);
	macroF1 = (double) (2*macroPrec*macroRecall)/(macroPrec + macroRecall);
	outputFile << std::fixed << std::setprecision(3) << macroAcc << " " << std::fixed << std::setprecision(3) << macroPrec << " " << std::fixed << std::setprecision(3) << macroRecall << " " << std::fixed << std::setprecision(3) << macroF1 << std::endl;
}

void NeuralNetwork::saveWeights(std::ofstream &outputFile) {
	//////std::cout << "numHidden = " << numHidden << ", numOutput = " << numOutput << std::endl;
	//////std::cout << "sizeofhidden = " << hiddenWeights.size() << ", sizeofoutput = " << outputWeights.size() << std::endl;
	
	outputFile << numInput << " " << numHidden << " " << numOutput << std::endl;
	for (int hiddenIter = 0; hiddenIter < numHidden; hiddenIter++) {
		for (int inputIter = 0; inputIter < numInput; inputIter++) {	
			outputFile << std::fixed << std::setprecision(3) << hiddenWeights[hiddenIter][inputIter] << " ";
		}
		outputFile << std::fixed << std::setprecision(3) << hiddenWeights[hiddenIter][numInput] << std::endl;
	}
	for (int outputIter = 0; outputIter < numOutput; outputIter++) {
		for (int hiddenIter = 0; hiddenIter < numHidden; hiddenIter++) {	
			outputFile << std::fixed << std::setprecision(3) << outputWeights[outputIter][hiddenIter] << " ";
		}
		outputFile << std::fixed << std::setprecision(3) << outputWeights[outputIter][numHidden] << std::endl;
	}
}

double NeuralNetwork::sigmoid(double inputVal) {
	return (1/(1+exp(-inputVal)));
}