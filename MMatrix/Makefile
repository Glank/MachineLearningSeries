CMP = g++
CFLAGS = -std=c++11 -Wall -Isrc

test_mmatrix: bin/mmatrix.o test/test_mmatrix.cc
	${CMP} -o test_mmatrix test/test_mmatrix.cc bin/mmatrix.o ${CFLAGS}

test_math: bin/mmatrix.o test/test_math.cc
	${CMP} -o test_math test/test_math.cc bin/mmatrix.o ${CFLAGS}

test_jpeg_decoder: test/test_jpeg_decoder.cc
	${CMP} -o test_jpeg_decoder test/test_jpeg_decoder.cc ${CFLAGS}

test_audio: bin/audio.o test/test_audio.cc bin/mmatrix.o
	${CMP} -o test_audio test/test_audio.cc bin/audio.o bin/mmatrix.o ${CFLAGS}

wav_example: src/wav_example.cc bin/audio.o bin/mmatrix.o
	echo "Reached"
	${CMP} -o wav_example src/wav_example.cc bin/audio.o bin/mmatrix.o ${CFLAGS}

bin/:
	mkdir bin

bin/mmatrix.o: bin/ src/mmatrix.cc src/mmatrix.h
	${CMP} -c src/mmatrix.cc -o bin/mmatrix.o ${CFLAGS}

bin/audio.o: bin/ src/audio.cc src/audio.h
	${CMP} -c src/audio.cc -o bin/audio.o ${CFLAGS}

clean_swp:
	-rm src/.*.swp
	-rm test/.*.swp

clean:
	-rm test_mmatrix
	-rm test_math
	-rm test_jpeg_decoder
	-rm test_audio
	-rm wav_example
	-rm -r bin
