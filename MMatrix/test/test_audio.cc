#include "audio.h"

#include <cmath>
#include <chrono> 
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std::chrono; 
using namespace audio;
using std::rand;

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
      exit(1);
    }
  }
}

void TestFourierTransform() {
  std::cout << "Testing Fourier Transform..." << std::endl;

  int sampleRate = 44100;
  // Frequencies A4 and C4
  mmatrix::DenseMMatrix frequencies({2});
  frequencies.set(0, 2*pi*440/sampleRate); // A4
  frequencies.set(1, 2*pi*262/sampleRate); // C4

  mmatrix::DenseMMatrix samples({sampleRate*4});
  for (int i = 0; i < sampleRate*4; i++) {
    if (i < sampleRate*2) {
      samples.set(i, 0.5*std::sin(frequencies.get(0)*i+0.73));
    } else {
      samples.set(i, 0.5*std::sin(frequencies.get(1)*i-0.17));
    }
  }

  int window = sampleRate/100;
  int nf = frequencies.shape()[0];
  int nt = samples.shape()[0]-window;
  mmatrix::DenseMMatrix ft({nf, nt});
  
  internal::FourierTransform(&samples, &frequencies, window, &ft);

  /*
  for (int i = -window; i < window; i+=10) {
    std::cout << ft.get({0, i+nt/2}) << "  " << ft.get({1,i+nt/2}) << std::endl;
  }
  // */

  if (ft.get({0, nt/2-window}) < ft.get({1, nt/2-window})) {
    std::cerr << "Should show A4 > C4 before halfway point." << std::endl;
    exit(1);
  }
  if (ft.get({0, nt/2+window}) > ft.get({1, nt/2+window})) {
    std::cerr << "Should show A4 < C4 after halfway point." << std::endl;
    exit(1);
  }
}

void TestFourierDifference() {
  std::cout << "Testing Fourier Difference..." << std::endl;

  int sampleRate = 8000;
  // Frequencies A4 and C4
  constexpr int A4 = 12*4;
  constexpr int C4 = 12*3+3;
  constexpr int E4 = C4+4;
  auto frequencies = internal::GetFrequencyDomain(sampleRate);

  // Sample 1 = A4 then C4 with E4
  mmatrix::DenseMMatrix samples1({sampleRate*2});
  for (int i = 0; i < sampleRate*2; i++) {
    double v = 0.5*std::sin(frequencies->get(E4)*i+0.05);
    if (i < sampleRate*1.5) {
      v += 0.5*std::sin(frequencies->get(A4)*i+0.73);
    } else {
      v += 0.5*std::sin(frequencies->get(C4)*i+0.17);
    }
    samples1.set(i, v);
  }

  // Sample 2 = A4 and E4
  mmatrix::DenseMMatrix samples2({sampleRate*2});
  for (int i = 0; i < sampleRate*2; i++) {
    double v = 0.5*std::sin(frequencies->get(E4)*i+0.69);
    v += 0.5*std::sin(frequencies->get(A4)*i+0.21);
    samples2.set(i, v);
  }

  // Sample 3 = C4 and E4
  mmatrix::DenseMMatrix samples3({sampleRate*2});
  for (int i = 0; i < sampleRate*2; i++) {
    double v = 0.5*std::sin(frequencies->get(E4)*i-0.75);
    v += 0.5*std::sin(frequencies->get(C4)*i+0.35);
    samples3.set(i, v);
  }

  int window = sampleRate/100;

  double d12 = internal::GetFourierDifference(&samples1, &samples2, frequencies.get(), window);
  double d13 = internal::GetFourierDifference(&samples1, &samples3, frequencies.get(), window);
  double d23 = internal::GetFourierDifference(&samples2, &samples3, frequencies.get(), window);

  /*
  {
    std::ofstream f("sample1.wav", std::ios::binary);
    WriteWavFile(f, sampleRate, 2, &samples1);
  }
  {
    std::ofstream f("sample2.wav", std::ios::binary);
    WriteWavFile(f, sampleRate, 2, &samples2);
  }
  {
    std::ofstream f("sample3.wav", std::ios::binary);
    WriteWavFile(f, sampleRate, 2, &samples3);
  }
  */

  if (d12 > d13) {
    std::cerr << "Expected d12 < d13" << std::endl;
    exit(1);
  }
  if (d12 > d23) {
    std::cerr << "Expected d12 < d23" << std::endl;
    exit(1);
  }
  if (d13 > d23) {
    std::cerr << "Expected d13 < d23" << std::endl;
    exit(1);
  }
}

void TestDerivFourierTransform() {
  std::cout << "Testing Deriv Fourier Transform..." << std::endl;

  int sampleRate = 8000;
  // Frequencies A4 and C4
  constexpr int A4 = 12*4;
  constexpr int E4 = 12*3+7;
  auto frequencies = internal::GetFrequencyDomain(sampleRate);

  mmatrix::DenseMMatrix samples({sampleRate});
  for (int i = 0; i < sampleRate; i++) {
    double v = 0.5*std::sin(frequencies->get(E4)*i-0.75);
    v += 0.5*std::sin(frequencies->get(A4)*i+0.35);
    samples.set(i, v);
  }

  int window = sampleRate/20;
  mmatrix::DenseMMatrix ft({frequencies->shape()[0], samples.shape()[0]-window});
  internal::FourierTransform(&samples, frequencies.get(), window, &ft);
  
  mmatrix::SparseMMatrix dft(mmatrix::Concat(ft.shape(), samples.shape()));

  auto start = high_resolution_clock::now();
  internal::DerivFourierTransorm(&samples, frequencies.get(), window, &ft, &dft);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  std::cout << duration.count() << std::endl; 
  // Before Opt:     83125628
  // After Opt:      24131967
  // Larger Window: 137323499

  int errors = 0;
  bool good = false;
  for (int ind = 0; ind < samples.shape()[0]; ind++) {
    int nonZeros = 0;
    for (int dep = 0; dep < ft.shape()[0]*ft.shape()[1]; dep++) {
      if (dft.get(dep+ft.shape()[0]*ft.shape()[1]*ind) != 0)
        nonZeros++;
    }
    if (nonZeros == 0) {
      if (good || errors == 0) {
        std::cerr << "Ind. var has all zeros in deriv start: " << ind << std::endl;
        good = false;
      }
      errors++;
    } else {
      if (!good) {
        std::cout << "good start: " << ind << std::endl;
        good = true;
      }
    }
  }

  mmatrix::DenseMMatrix samples2({sampleRate});
  mmatrix::DenseMMatrix ft2(ft.shape());
  std::srand(314);
  double epsilon = 0.01;
  for (int i = 0; i < window+20; i++) {
    mmatrix::Copy(&samples, &samples2);
    int dep_f = 12*4; //std::rand()%(ft.shape()[0]);
    int dep_i = i; //std::rand()%(ft.shape()[1]);
    int dep = dep_f+ft.shape()[0]*dep_i;
    int ind = window+10; // std::rand()%(samples.shape()[0]);
    double before = ft.get(dep);
    samples2.set(ind, samples2.get(ind)+epsilon);
    internal::FourierTransform(&samples2, frequencies.get(), window, &ft2);
    double after = ft2.get(dep);
    double approx_deriv = (after-before)/epsilon;
    double act_deriv = dft.get(dep+ft.shape()[0]*ft.shape()[1]*ind);
    double err = std::abs(approx_deriv-act_deriv);
    /*
    std::cerr << "err: " << err << std::endl;
    std::cerr << "Act=" << act_deriv << "    Approx=" << approx_deriv << std::endl;
    std::cerr << "dep_f:" << dep_f << "    dep_i: " << dep_i << "    ind:" << ind << std::endl;
    // */
    //*
    if (err > 0.01) {
      std::cerr << "Mismatch Act=" << act_deriv << "    Approx=" << approx_deriv << std::endl;
      std::cerr << "dep_f:" << dep_f << "    dep_i: " << dep_i << "    ind:" << ind << std::endl;
      exit(1);
    }
    // */
  }

}

void TestDerivFourierDifference() {
  std::cout << "Testing Deriv Fourier Difference..." << std::endl;

  int sampleRate = 8000;
  // Frequencies A4 and C4
  constexpr int A4 = 12*4;
  constexpr int E4 = 12*3+7;
  auto frequencies = internal::GetFrequencyDomain(sampleRate);

  mmatrix::DenseMMatrix samples1({sampleRate});
  for (int i = 0; i < sampleRate; i++) {
    double v = 0.5*std::sin(frequencies->get(E4)*i-0.75);
    v += 0.5*std::sin(frequencies->get(A4)*i+0.35);
    samples1.set(i, v);
  }

  std::srand(314);
  mmatrix::DenseMMatrix samples2({sampleRate});
  for (int i = 0; i < sampleRate; i++) {
    double v = 2*(rand()/static_cast<double>(RAND_MAX))-1;
    samples2.set(i, v);
  }

  int window = sampleRate/20;

  mmatrix::DenseMMatrix ft1({frequencies->shape()[0], samples1.shape()[0]-window});
  mmatrix::DenseMMatrix ft2({frequencies->shape()[0], samples2.shape()[0]-window});
  mmatrix::SparseMMatrix dft(mmatrix::Concat(ft2.shape(), samples2.shape()));
  mmatrix::DenseMMatrix dfd(samples2.shape());

  constexpr double learningRate = 1;

  {
    std::ofstream f("goal.wav", std::ios::binary);
    WriteWavFile(f, sampleRate, 2, &samples1);
  }
  {
    std::ofstream f("start.wav", std::ios::binary);
    WriteWavFile(f, sampleRate, 2, &samples2);
  }

  for (int i = 0; i < 30; i++) {
    std::cout << "Itteration " << i << std::endl;

    auto start = high_resolution_clock::now();
    internal::FourierTransform(&samples1, frequencies.get(), window, &ft1);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "ft1: " << duration.count() << std::endl; 

    start = high_resolution_clock::now();
    internal::FourierTransform(&samples2, frequencies.get(), window, &ft2);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "ft2: " << duration.count() << std::endl; 

    start = high_resolution_clock::now();
    internal::DerivFourierTransorm(&samples2, frequencies.get(), window, &ft2, &dft);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "dft: " << duration.count() << std::endl; 

    start = high_resolution_clock::now();
    mmatrix::SubFrom(&ft1, &ft2);
    mmatrix::Multiply(2, &ft2, &dft, &dfd);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "dfd: " << duration.count() << std::endl; 


    double sz = 0;
    for (int j = 0; j < dfd.shape()[0]; j++) {
      if (j%100 == 0) {
        std::cout << j  << ": " <<  dfd.get(j) << std::endl;
      }
      sz += dfd.get(j)*dfd.get(j);
    }
    sz = std::sqrt(sz);
    std::cout << "deriv size: " << sz << std::endl;
    mmatrix::Elementwise([=](mmatrix::MMFloat x){return learningRate*x/sz;}, &dfd, &dfd);
  
    start = high_resolution_clock::now();
    mmatrix::SubFrom(&dfd, &samples2);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "update: " << duration.count() << std::endl; 

    std::stringstream ss;
    ss << "itteration" << i << ".wav";
    std::ofstream f(ss.str(), std::ios::binary);
    WriteWavFile(f, sampleRate, 2, &samples2);
  }
}

int main() {
  //TestWriteRead();
  //TestFourierTransform();
  //TestFourierDifference();
  TestDerivFourierTransform();
  //TestDerivFourierDifference();
  std::cout << "All tests pass." << std::endl;
  return 0;
}
