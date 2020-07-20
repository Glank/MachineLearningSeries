#ifndef AUDIO_H
#define AUDIO_H

#include "mmatrix.h"

#include <cstdint>
#include <iostream>
#include <vector>

namespace audio {

typedef struct WavHdr {
  /* RIFF Chunk Descriptor */
  uint8_t         RIFF[4];        // RIFF Header Magic header
  uint32_t        ChunkSize;      // RIFF Chunk Size
  uint8_t         WAVE[4];        // WAVE Header
  /* "fmt" sub-chunk */
  uint8_t         fmt[4];         // FMT header
  uint32_t        Subchunk1Size;  // Size of the fmt chunk
  uint16_t        AudioFormat;    // Audio format 1=PCM,6=mulaw,7=alaw,     257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM
  uint16_t        NumOfChan;      // Number of channels 1=Mono 2=Stereo
  uint32_t        SamplesPerSec;  // Sampling Frequency in Hz
  uint32_t        bytesPerSec;    // bytes per second
  uint16_t        blockAlign;     // 2=16-bit mono, 4=16-bit stereo
  uint16_t        bitsPerSample;  // Number of bits per sample
  /* "data" sub-chunk */
  uint8_t         Subchunk2ID[4]; // "data"  string
  uint32_t        Subchunk2Size;  // Sampled data length
} WavHdr;

bool isBigEndian();

void ReadWavFile(std::istream& file, WavHdr* hdr, std::vector<double>* samples);
std::unique_ptr<mmatrix::DenseMMatrix> ReadWavFile(std::istream& file, WavHdr* hdr);

void WriteWavFile(std::ostream& file, uint32_t sampleRate, int bytesPerSample,
    const std::vector<double>& samples);
void WriteWavFile(std::ostream& file, uint32_t sampleRate, int bytesPerSample,
    const mmatrix::MMatrixInterface* samples);

namespace internal {

// Returs a vector of 2*pi/sampleRate adjusted frequency constants.
std::unique_ptr<mmatrix::DenseMMatrix> GetFrequencyDomain(uint32_t sampleRate);

// Does a rolling Fourier transform with a window of `window`
// `frequencies` should be a list of 2*pi/sampleRate adjusted frequency constants.
// The complex results are reduced to the reals by taking the size of their complex vector.
// out.shape = <frequency, time offset>
void FourierTransform(const mmatrix::MMatrixInterface* samples,
    const mmatrix::MMatrixInterface* frequencies,
    int window, mmatrix::MMatrixInterface* out);

double GetFourierDifference(
    const mmatrix::MMatrixInterface* samples1,
    const mmatrix::MMatrixInterface* samples2,
    const mmatrix::MMatrixInterface* frequencies,
    int window);

// Returns the derivative of a fourier transform by samples.
// If s is the number of samples, f the number of frequencies, and w is window,
// then the shape of the output should be <f, s-w, s>
// It is also recommended that output be a sparse mmatrix.
void DerivFourierTransorm(
    const mmatrix::MMatrixInterface* samples,
    const mmatrix::MMatrixInterface* frequencies,
    int window,
    const mmatrix::MMatrixInterface* ft,
    mmatrix::MMatrixInterface* out);

}  // namespace internal


}  // namespace audio

#endif // AUDIO_H
