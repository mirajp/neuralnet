netTrainer.out: neuralNet.o netTrainer.o
	g++ $^ -o $@

netTester.out: neuralNet.o netTester.o
	g++ $^ -o $@

neuralNet.o: neuralNet.cpp neuralNet.h
	g++ -c $^

netTrainer.o: netTrainer.cpp
	g++ -c $^
	
netTester.o: netTester.cpp
	g++ -c $^

clean:
	rm *.out *.o *.stackdump *~
