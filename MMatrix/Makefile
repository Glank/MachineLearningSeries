CFLAGS = -std=c++11 -Wall

test_mmatrix: mmatrix.o test_mmatrix.cc
	g++ -g -o test_mmatrix test_mmatrix.cc mmatrix.o ${CFLAGS}

mmatrix.o: mmatrix.cc mmatrix.h
	g++ -g -c mmatrix.cc ${CFLAGS}

clean:
	-rm test_mmatrix
	-rm *.o
