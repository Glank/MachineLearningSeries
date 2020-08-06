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
  // TODO reqork to take advantage of sparseness of dft
  for (long ind = 0; ind < samples.shape()[0]; ind++) {
    int nonZeros = 0;
    for (long dep = 0; dep < ft.shape()[0]*ft.shape()[1]; dep++) {
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
  std::cout << "Empty row check end." << std::endl;

  mmatrix::DenseMMatrix samples2({sampleRate});
  mmatrix::DenseMMatrix ft2(ft.shape());
  std::srand(314);
  double epsilon = 0.01;
  for (int i = 0; i < window+20; i++) {
    mmatrix::Copy(&samples, &samples2);
    long dep_f = 12*4; //std::rand()%(ft.shape()[0]);
    long dep_i = i; //std::rand()%(ft.shape()[1]);
    long dep = dep_f+ft.shape()[0]*dep_i;
    long ind = window+10; // std::rand()%(samples.shape()[0]);
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
  std::unique_ptr<mmatrix::DenseMMatrix> dfd(new mmatrix::DenseMMatrix(samples2.shape()));
  std::unique_ptr<mmatrix::DenseMMatrix> dfdPrev(new mmatrix::DenseMMatrix(samples2.shape()));
  mmatrix::DenseMMatrix delta(samples2.shape());
  std::unique_ptr<mmatrix::DenseMMatrix> tmp;
  mmatrix::DenseMMatrix dgdl({});

  double learningRate = 0.000001;
  double maxLearningRate = 0.0001;
  double learningScaleFactor = 1.2;

  {
    std::ofstream f("data2/goal.wav", std::ios::binary);
    WriteWavFile(f, sampleRate, 2, &samples1);
  }
  {
    std::ofstream f("data2/start.wav", std::ios::binary);
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
    // swap dfdPrev and dfd
    tmp = std::move(dfdPrev);
    dfdPrev = std::move(dfd);
    dfd = std::move(tmp);
    mmatrix::SubFrom(&ft1, &ft2);
    mmatrix::Multiply(2, &ft2, &dft, dfd.get());
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "dfd: " << duration.count() << std::endl; 


    // Update learning rate
    if (i > 0) {
      start = high_resolution_clock::now();
      mmatrix::Multiply(1, dfdPrev.get(), dfd.get(), &dgdl);
      if (dgdl.get(0) > 0 && learningRate < maxLearningRate) {
        learningRate *= learningScaleFactor;
      } else {
        learningRate /= learningScaleFactor;
      }
      std::cout << "learningRate updated to:" << learningRate << std::endl;
      stop = high_resolution_clock::now();
      duration = duration_cast<microseconds>(stop - start);
      std::cout << "lr update: " << duration.count() << std::endl; 
    }

    start = high_resolution_clock::now();
    double sz = 0;
    for (int j = 0; j < dfd->shape()[0]; j++) {
      sz += dfd->get(j)*dfd->get(j);
    }
    sz = std::sqrt(sz);
    std::cout << "deriv size: " << sz << std::endl;
    mmatrix::Elementwise([=](mmatrix::MMFloat x){return learningRate*x;}, dfd.get(), &delta);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "rescale: " << duration.count() << std::endl; 
  
    start = high_resolution_clock::now();
    mmatrix::SubFrom(&delta, &samples2);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "update: " << duration.count() << std::endl; 

    std::stringstream ss;
    ss << "data2/itteration" << i << ".wav";
    std::ofstream f(ss.str(), std::ios::binary);
    WriteWavFile(f, sampleRate, 2, &samples2);
  }
}

void fft(std::vector<double> s, int N, int a, int b, std::vector<double>* outR, std::vector<double>* outI) {
  if (N == 1) {
    outR->push_back(s[b]);
    outI->push_back(0);
    return;
  }
  int kOff = outR->size();
  // Calculate even parts
  fft(s, N/2, 2*a, b, outR, outI);
  // Calculate odd parts
  fft(s, N/2, 2*a, a+b, outR, outI);
  for (int k = 0; k < N/2; k++) {
    double c = std::cos(2*pi*k/N);
    double s = std::sin(2*pi*k/N);
    double evenR = (*outR)[k+kOff];
    double evenI = (*outI)[k+kOff];
    double oddR = (*outR)[k+N/2+kOff];
    double oddI = (*outI)[k+N/2+kOff];
    (*outR)[k+kOff] = evenR+c*oddR-s*oddI;
    (*outI)[k+kOff] = evenI+s*oddR+c*oddI;
    (*outR)[k+N/2+kOff] = evenR-c*oddR+s*oddI;
    (*outI)[k+N/2+kOff] = evenI-s*oddR-c*oddI;
    //outR->push_back(evenR[km]+c*oddR[km]-s*oddI[km]);
    //outI->push_back(evenI[km]+s*oddR[km]+c*oddI[km]);
  }
}

void sft(std::vector<double> s, int N, int a, int b, std::vector<double>* outR, std::vector<double>* outI) {
  for (int k = 0; k < N; k++) {
    double real = 0, imag = 0;
    for (int i = 0; i < N; i++) {
      real += std::cos(2*pi*i*k/N)*s[a*i+b];
      imag += std::sin(2*pi*i*k/N)*s[a*i+b];
    }
    outR->push_back(real);
    outI->push_back(imag);
  }
}

void TestFFT() {
  std::vector<double> samples;
  int N = 4096;
  std::srand(314);
  // 44100 1/second
  int sampleRate = 44100; //44100;
  // peaks/second
  // A0 = 27.5
  // C8 = 4186
  double realF = 440; // 27.5; // 27.5*std::pow(2, 1/12.0);
  // peaks/sample
  double sampleF = realF/sampleRate;
  // fft bucket for frequency.
  double kForSampleF = sampleF*N;

  std::cout << "Expected k: " << kForSampleF << std::endl;
  for (int i = 0; i < N; i++) {
    double w =  0.5*(1-std::cos(2*pi*i/N));  // hann window
    samples.push_back(w*(0.5*std::sin(2*pi*i*kForSampleF/N) + 0.01*(std::rand()/static_cast<double>(RAND_MAX)-0.5)));
  }
  
  {
    std::ofstream f("sample.wav", std::ios::binary);
    WriteWavFile(f, sampleRate, 2, samples);
  }

  std::vector<double> fftR, fftI;
  fft(samples, N, 1, 0, &fftR, &fftI);
  std::vector<double> sftR, sftI;
  sft(samples, N, 1, 0, &sftR, &sftI);

  /*
  for (int i = 0; i < N; i++) {
    std::cout << fftR[i] << '\t' << fftI[i] << '\t';
    std::cout << sftR[i] << '\t' << sftI[i] << std::endl;
  }
  exit(0);
  // */
  
  int maxK = -1;
  double maxAmp2 = -1;
  for (int k = 0; k < N/2; k++) {
    double amp2 = fftR[k]*fftR[k]+fftI[k]*fftI[k];
    //std::cout << k << ": " << amp2 << std::endl;
    if (amp2 > maxAmp2) {
      maxK = k;
      maxAmp2 = amp2;
    }

    double realDiff = std::abs(fftR[k]-sftR[k]);
    double imagDiff = std::abs(fftI[k]-sftI[k]);
    if (realDiff > 0.1) {
      std::cerr << "Real mismatch: " << k << std::endl;
      exit(1);
    }
    if (imagDiff > 0.1) {
      std::cerr << "Imaginary mismatch: " << k << std::endl;
      exit(1);
    }
  }
  std::cout << "Found k: " << maxK << std::endl;
  std::cout << "Found amp: " << maxAmp2 << std::endl;
  // 27.5 * 2^(n/12) = f
  // N*f/sr = k
  // f = k*sr/N
  // n = 12*ln(f/27.5)/ln(2)
  // n = 12*ln(k*sr/(N*27.5))/ln(2)
  double freq = maxK*sampleRate/N;
  std::cout << "Frequency: " << freq << std::endl;
  double note = 12*std::log(freq/27.5)/std::log(2);
  
  std::cout << "Note: " << note << std::endl;
  int noteInt = static_cast<int>(std::round(note))%12;
  std::cout << "Note (int): " << noteInt << std::endl;

  mmatrix::DenseMMatrix a({3,3});
  for (int c = 0; c < 3; c++) {
    for (int r = 0; r < 3; r++) {
      a.set({r,c}, std::pow((maxK-1+r), 2-c));
    }
  }
  mmatrix::DenseMMatrix aInv({3,3});
  mmatrix::Invert(1, &a, &aInv);
  mmatrix::DenseMMatrix x({3});
  mmatrix::DenseMMatrix b({3});
  int k = maxK;
  for (int i = 0; i < 3; i++) {
    double amp = fftR[k-1+i]*fftR[k-1+i]+fftI[k-1+i]*fftI[k-1+i];
    b.set(i, amp);
  }
  mmatrix::Multiply(1, &aInv, &b, &x);

  for (int i = 0; i < 3; i++) {
    std::cout << x.get(i) << std::endl;
  }
  double validateAmp = x.get(0)*maxK*maxK+x.get(1)*maxK+x.get(2);
  std::cout << "Validate Amp: " << validateAmp << std::endl;
  
  double adjustedK = -x.get(1)/(2*x.get(0));
  std::cout << "Adjusted K: " << adjustedK << std::endl;

  double adjFreq = adjustedK*sampleRate/N;
  std::cout << "Adjusted Frequency: " << adjFreq << std::endl;
  double adjNote = 12*std::log(adjFreq/27.5)/std::log(2);
  
  std::cout << "Adjusted Note: " << adjNote << std::endl;
  int adjNoteInt = static_cast<int>(std::round(adjNote))%12;
  std::cout << "Adjusted Note (int): " << adjNoteInt << std::endl;

  {
    std::ofstream f("fft.csv", std::ios::binary);
    f << "k\tfreq\tR\tI\tAmp2" << std::endl;
    for (int i = 0; i < N; i++) {
      double freq = i*sampleRate/static_cast<double>(N);
      f << i << '\t'
        << freq << '\t'
        << fftR[i] << '\t'
        << fftI[i] << '\t'
        << (fftR[i]*fftR[i]+fftI[i]*fftI[i]) << std::endl;
    }
  }
}

// Frequency in hz
// window in seconds
void partialFT(double frequency, double sampleRate, double window, const std::vector<double>& samples, std::vector<double>* out) {
  double real = 0, imag = 0;
  double f = frequency/sampleRate;
  int w = static_cast<int>(std::ceil((window*sampleRate)-1)/2)*2+1;
  // std::cout << w << std::endl;

  for (int i = 0; i < w/2; i++) {
    real += samples[i] * std::cos(2*pi*f*i);
    imag += samples[i] * std::sin(2*pi*f*i);
  }

  for (int i = 0; i < samples.size(); i++) {
    int head = i+w/2+1;
    int tail = i-w/2;
    if (head < samples.size()) {
      real += samples[head] * std::cos(2*pi*f*head);
      imag += samples[head] * std::sin(2*pi*f*head);
    }
    if (tail >= 0) {
      real -= samples[tail] * std::cos(2*pi*f*tail);
      imag -= samples[tail] * std::sin(2*pi*f*tail);
    }
    out->push_back(std::sqrt(real*real+imag*imag));
  }
}

void pushFrequency(double frequency, double sampleRate, double seconds, std::vector<double>* samples) {
  double sampleF = frequency/sampleRate;
  double nSamples = sampleRate*seconds;
  for (int i = 0; i < nSamples; i++) {
    samples->push_back(0.5*std::sin(2*pi*i*sampleF));
  }
}

void partialFT(const std::vector<double>& frequencies, double sampleRate, const std::vector<double>& samples, mmatrix::MMatrixInterface* out) {
  std::vector<double> partial;
  partial.reserve(samples.size());
  for (int f = 0; f < frequencies.size(); f++) {
    double freq = frequencies[f];
    partial.clear();
    partial.reserve(samples.size());
    partialFT(freq, sampleRate, 5.0/freq, samples, &partial);
    for (int s = 0; s < samples.size(); s++) {
      out->set({f, s}, partial[s]);
    }
  }
}

void TestPartialFT() {
  std::vector<double> samples;
  std::srand(314);
  // 44100 1/second
  int sampleRate = 44100; //44100;
  // peaks/second
  // A0 = 27.5
  // C8 = 4186
  //double a4 = 440; // 27.5; // 27.5*std::pow(2, 1/12.0);
  double c4 = 261.63;
  double d4 = 293.66;
  double e4 = 329.63;

  std::vector<double> notes{e4, d4, c4, d4, e4};
  for (int i = 0; i < notes.size(); i++) {
    pushFrequency(notes[i], sampleRate, 0.25, &samples);
  }

  {
    std::ofstream f("sample.wav", std::ios::binary);
    WriteWavFile(f, sampleRate, 2, samples);
  }

  mmatrix::DenseMMatrix out({3,static_cast<int>(samples.size())});
  
  std::vector<double> frequencies{c4,d4,e4};
  std::vector<std::string> fnames{"c4", "d4", "e4"};
  partialFT(frequencies, sampleRate, samples, &out);

  int lastMax = -1;
  int maxStart = 0;
  for (int i = 0; i < samples.size(); i++) {
    //*
    int max = -1;
    double maxValue = -1;
    for (int f = 0; f < frequencies.size(); f++) {
      if (out.get({f, i}) > maxValue) {
        maxValue = out.get({f, i});
        max = f;
      }
    }
    if (max != lastMax) {
      if (i - maxStart > sampleRate*0.01) {
        std::cout << fnames[lastMax] << std::endl;;
      }
      // std::cout << i << ": " << fnames[max] << " " << maxValue << "  " << (i/static_cast<double>(sampleRate)) << "  " << (i-maxStart)/static_cast<double>(sampleRate) << std::endl;
      //std::cout << out.get({0, i}) << "  " << out.get({1,i}) << "  " << out.get({2,i}) << std::endl;
      maxStart = i;
    }
    lastMax = max;
    // */
    /*
    if (i % 100 == 0)
      std::cout << i << ": " << out.get({2,i}) << std::endl;
    // */
  }
}

void SimulateDrum() {
  std::vector<double> samples;
  std::srand(314);
  // 44100 1/second
  int sampleRate = 44100; //44100;

  for (int b = 0; b < 10; b++) {
    for (int i = 0; i < sampleRate/4; i++) {
      double s = 0;
      s += (std::rand()/static_cast<double>(RAND_MAX))*std::exp(-i*0.01);
      for (double f : {440, 480, 520}) {
        s += 0.5*std::sin(2*pi*f/sampleRate*i)*std::exp(-i*0.001);
      }
      samples.push_back(s);
    }
  }

  {
    std::ofstream f("sample.wav", std::ios::binary);
    WriteWavFile(f, sampleRate, 2, samples);
  }
}

void SimulateCymbal() {
  std::vector<double> samples;
  std::srand(314);
  // 44100 1/second
  int sampleRate = 44100; //44100;

  // See: https://www.soundonsound.com/techniques/synthesizing-realistic-cymbals

  // Top Curve
  std::function<double(double)> topCurve = [&](double t_){
    constexpr int k = 7;
    constexpr double tPts[k] = {-4, -3.9, -3.5, -2, -1, 0, 1};
    constexpr double hPts[k] = {3.5, 3.8, 3.1, 4.1, 4.5, 4.1, 2};
    double t = std::log(t_)/std::log(10);
    int i;
    for (i = 0; i < k && tPts[i] < t; i++);
    if (i == 0) {
      i = 1;
    }
    if (i == k) {
      i = k-1;
    }
    double t1 = tPts[i-1];
    double t2 = tPts[i];
    double h1 = hPts[i-1];
    double h2 = hPts[i];
    double dt = t2-t1;
    double dh = h2-h1;
    double tp = (t-t1)/dt;
    double h = tp*dh+h1;
    return std::pow(10,h);
  };
  
  {
    std::ofstream f("top_curve.csv");
    f << "t\th" << std::endl;
    for (double t = -4.1; t < 1; t+=0.1) {
        f << t << '\t'
          << std::log(topCurve(std::pow(10,t)))/std::log(10) << std::endl;
    }
  }

  // Bottom Curve
  std::function<double(double)> bottomCurve = [&](double t_){
    constexpr int k = 5;
    constexpr double tPts[k] = {-4, -2.5, -1.1, 0, 1};
    constexpr double hPts[k] = {1, 2.5, 2.9, 2.1, 0.8};
    double t = std::log(t_)/std::log(10);
    int i;
    for (i = 0; i < k && tPts[i] < t; i++);
    if (i == 0) {
      i = 1;
    }
    if (i == k) {
      i = k-1;
    }
    double t1 = tPts[i-1];
    double t2 = tPts[i];
    double h1 = hPts[i-1];
    double h2 = hPts[i];
    double dt = t2-t1;
    double dh = h2-h1;
    double tp = (t-t1)/dt;
    double h = tp*dh+h1;
    return std::pow(10,h);
  };
  
  {
    std::ofstream f("curves.csv");
    f << "t\ttop\tbottom" << std::endl;
    for (double t = -4.1; t < 1; t+=0.1) {
        f << t << '\t'
          << std::log(topCurve(std::pow(10,t)))/std::log(10) << '\t'
          << std::log(bottomCurve(std::pow(10,t)))/std::log(10) << std::endl;
    }
  }

  double df = 1.05;
  for (int b = 0; b < 10; b++) {
    for (int i = 0; i < sampleRate/4; i++) {
      double s = 0;
      double t = i/static_cast<double>(sampleRate);
      double top = topCurve(t);
      double bottom  = bottomCurve(t);
      for (double h = 10; h < top; h*=df) {
        if (h < bottom) continue;
        //double h = (top-bottom)*(f/static_cast<double>(nf))+bottom;
        s += 0.1*std::sin(2*pi*h/sampleRate*i)*std::exp(-i*0.0005);
        s += 0.05*(std::rand()/static_cast<double>(RAND_MAX)-0.5)*std::exp(-i*0.001);
      }
      samples.push_back(s);
    }
  }


  {
    std::ofstream f("sample.wav", std::ios::binary);
    WriteWavFile(f, sampleRate, 2, samples);
  }
}

int main() {
  //TestWriteRead();
  //TestFourierTransform();
  //TestFourierDifference();
  //TestDerivFourierTransform();
  //TestDerivFourierDifference();
  //TestFFT();
  //TestPartialFT();
  //SimulateDrum();
  SimulateCymbal();
  std::cout << "All tests pass." << std::endl;
  return 0;
}
