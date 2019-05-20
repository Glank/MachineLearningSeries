CFLAGS = -std=c++11 -Wall

genetic: genetic.o
	g++ -g -o genetic genetic.o ${CFLAGS}

genetic.o: genetic.cc
	g++ -g -c genetic.cc ${CFLAGS}

clean:
	-rm genetic
	-rm *.o
