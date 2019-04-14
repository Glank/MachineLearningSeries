CFLAGS = -std=c++11 -Wall

smiley: smiley.o
	g++ -g -o smiley smiley.o ${CFLAGS}

smiley.o: smiley.cc
	g++ -g -c smiley.cc ${CFLAGS}

clean:
	-rm smiley
	-rm smiley.o
