CMP = g++
CFLAGS = -std=c++11 -Wall -Isrc

test_mmatrix: bin/mmatrix.o test/test_mmatrix.cc
	${CMP} -o test_mmatrix test/test_mmatrix.cc bin/mmatrix.o ${CFLAGS}

test_math: bin/mmatrix.o test/test_math.cc
	${CMP} -o test_math test/test_math.cc bin/mmatrix.o ${CFLAGS}

test_jpeg_decoder: test/test_jpeg_decoder.cc
	${CMP} -o test_jpeg_decoder test/test_jpeg_decoder.cc ${CFLAGS}

bin/:
	mkdir bin

bin/mmatrix.o: bin/ src/mmatrix.cc src/mmatrix.h
	${CMP} -c src/mmatrix.cc -o bin/mmatrix.o ${CFLAGS}

clean:
	-rm test_mmatrix
	-rm test_jpeg_decoder
	-rm -r test_mmatrix.dSYM
	-rm -r bin
