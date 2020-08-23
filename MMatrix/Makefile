CMP = g++
CFLAGS = -std=c++11 -Wall -Isrc

all: test_mmatrix test_math test_covering test_jpeg_decoder test_audio wav_example

test_mmatrix: bin/mmatrix.o test/test_mmatrix.cc
	${CMP} -o test_mmatrix test/test_mmatrix.cc bin/mmatrix.o ${CFLAGS}

test_math: bin/mmatrix.o bin/mmatrix_test_utils.o test/test_math.cc src/math_functions.h
	${CMP} -o test_math test/test_math.cc bin/mmatrix.o bin/mmatrix_test_utils.o ${CFLAGS}

test_covering: bin/mmatrix.o bin/covering.o bin/mmatrix_test_utils.o test/test_covering.cc
	${CMP} -o test_covering test/test_covering.cc bin/mmatrix.o bin/covering.o bin/mmatrix_test_utils.o ${CFLAGS}

test_jpeg_decoder: test/test_jpeg_decoder.cc
	${CMP} -o test_jpeg_decoder test/test_jpeg_decoder.cc ${CFLAGS}

test_audio: bin/audio.o test/test_audio.cc bin/mmatrix.o bin/covering.o
	${CMP} -o test_audio test/test_audio.cc bin/audio.o bin/mmatrix.o bin/covering.o ${CFLAGS}

wav_example: src/wav_example.cc bin/audio.o bin/mmatrix.o
	${CMP} -o wav_example src/wav_example.cc bin/audio.o bin/mmatrix.o ${CFLAGS}

bin/:
	mkdir bin

bin/mmatrix.o: bin/ src/mmatrix.cc src/mmatrix.h
	${CMP} -c src/mmatrix.cc -o bin/mmatrix.o ${CFLAGS}

bin/audio.o: bin/ src/audio.cc src/audio.h
	${CMP} -c src/audio.cc -o bin/audio.o ${CFLAGS}

bin/covering.o: bin/mmatrix.o src/covering.cc src/covering.h src/math_functions.h
	${CMP} -c src/covering.cc -o bin/covering.o ${CFLAGS}

bin/mmatrix_test_utils.o: bin/mmatrix.o src/mmatrix_test_utils.cc src/mmatrix_test_utils.h
	${CMP} -c src/mmatrix_test_utils.cc -o bin/mmatrix_test_utils.o ${CFLAGS}

clean_swp:
	-rm src/.*.swp
	-rm test/.*.swp

clean:
	-rm test_audio
	-rm test_covering
	-rm test_jpeg_decoder
	-rm test_math
	-rm test_mmatrix
	-rm wav_example
	-rm -r bin
