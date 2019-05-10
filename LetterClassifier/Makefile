CFLAGS = -std=c++11 -Wall

test_mmatrix: mmatrix.o test_mmatrix.cc
	g++ -g -o test_mmatrix test_mmatrix.cc mmatrix.o ${CFLAGS}

genetic: genetic.o
	g++ -g -o genetic genetic.o ${CFLAGS}

genetic.o: genetic.cc
	g++ -g -c genetic.cc ${CFLAGS}

mmatrix.o: mmatrix.cc mmatrix.h
	g++ -g -c mmatrix.cc ${CFLAGS}

clean:
	-rm genetic
	-rm test_mmatrix
	-rm *.o
