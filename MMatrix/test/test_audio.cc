#include "audio.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace audio;

const double pi = std::acos(-1);

void TestWriteRead() {
  std::cout << "Testing Write & Read..." << std::endl;

  std::stringstream ss;

  // Write out a wave file to ss
  int sampleRate = 44100;
  std::vector<double> samplesIn;
  for (double t = 0; t < 10; t+=1.0/sampleRate) {
    // A4 frequency is 440
    samplesIn.push_back(0.5*std::sin(2*pi*440*t));
  }

  WriteWavFile(ss, sampleRate, 2, samplesIn);

  // Read the wave file in ss
  ss.seekg(0);
  WavHdr hdr;
  std::vector<double> samplesOut;
  ReadWavFile(ss, &hdr, &samplesOut);
  if (samplesOut.size() != samplesIn.size()) {
    std::cerr << "Unexpected samples count: " << samplesOut.size() << std::endl;
    exit(1);
  }

  // Verify rough sample consistency (will vary due to compression artifacts).
  for (int i = 0; i < 10; i++) {
    if (std::abs(samplesIn[i]-samplesOut[i]) > 0.0001) {
      std::cerr << "Sample divation out of bounds: " << i
          << " (" << samplesIn[i] << ", " << samplesOut[i] << ")"
          << std::endl;
      exit(0);
    }
  }
}

int main() {
  TestWriteRead();
  std::cout << "All tests pass." << std::endl;
  return 0;
}
