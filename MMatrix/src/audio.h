#ifndef AUDIO_H
#define AUDIO_H

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

void WriteWavFile(std::ostream& file, uint32_t sampleRate, int bytesPerSample,
    const std::vector<double>& samples);

namespace internal {

std::vector<double> GetFrequencyDomain(uint32_t sampleRate);

void fourier_transform(std::vector<double> samples, std::vector<double> frequencies, int delta);

}  // namespace internal


}  // namespace audio

#endif
