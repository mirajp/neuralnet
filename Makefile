netTrainer.out: neuralNet.o netTrainer.o
	g++ $^ -o $@

netTester.out: neuralNet.o netTester.o
	g++ $^ -o $@

neuralNet.o: neuralNet.cpp neuralNet.h
	g++ -c $^ -o $@

netTrainer.o: netTrainer.cpp
	g++ -c $^ -o $@
	
netTester.o: netTester.cpp
	g++ -c $^ -o $@

clean:
	rm *.out *.o *.stackdump *~
