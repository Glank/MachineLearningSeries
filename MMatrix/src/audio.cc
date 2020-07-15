#include "audio.h"

#include <limits>
#include <string.h>
#include <cmath>

namespace audio {


bool isBigEndian() {
    union {
        uint32_t i;
        char c[4];
    } bint = {0x01020304};

    return bint.c[0] == 1; 
}

void ReadWavFile(std::istream& file, WavHdr* hdr, std::vector<double>* samples) {
  if(isBigEndian()) {
    throw std::runtime_error("This library is not supported on big endian systems.");
  }

  // Get file size and read header.
  file.seekg(0, std::ios::end);
  int fsize = file.tellg();
  file.seekg(0);
  if (fsize < sizeof(WavHdr)) {
    throw std::runtime_error("Invalid file.");
  }
  file.read(reinterpret_cast<char*>(hdr), sizeof(WavHdr));
  if (hdr->AudioFormat != 1) {
    throw std::runtime_error("Unsupported wav compression format.");
  }

  // Calculate sample size from header.
  int chanSampleSize;
  if (hdr->bitsPerSample == 8 || hdr->bitsPerSample == 16) {
    chanSampleSize = hdr->bitsPerSample/8;
  } else {
    throw std::runtime_error("Unexpected bits per sample.");
  }
  int sampleSize = hdr->NumOfChan*chanSampleSize;

  // Validate header, file size and expected samples 
  // Subchunk2Size  == NumSamples * NumChannels * BitsPerSample/8
  // NumSamples == Subchunk2Size / sampleSize
  if (hdr->Subchunk2Size % sampleSize != 0) {
    throw std::runtime_error("Invalid Subchunk2Size in wav header.");
  }
  if (hdr->Subchunk2Size + sizeof(WavHdr) > fsize) {
    throw std::runtime_error("Invalid Subchunk2Size for wav file size.");
  }
  int expectedSamples = hdr->Subchunk2Size/sampleSize;
   
  // Read the samples
  samples->reserve(expectedSamples);
  union {
    uint8_t b8;
    int16_t b16;
  } channelSample;
  double sample;
  // This is a tremendiously inefficieint way to read a file.
  // TODO: buffer
  for (int i = 0; i < expectedSamples; i++) {
    sample = 0;
    for (int c = 0; c < hdr->NumOfChan; c++) {
      file.read(reinterpret_cast<char*>(&channelSample), chanSampleSize);
      if (chanSampleSize == 1) {
        sample += channelSample.b8/static_cast<double>(1<<7)-1.0;
      } else {
        sample += channelSample.b16/static_cast<double>(1<<15);
      }
    }
    sample /= hdr->NumOfChan;
    samples->push_back(sample);
  }
}

void WriteWavFile(std::ostream& file, uint32_t sampleRate, int bytesPerSample, const std::vector<double>& samples) {
  if (bytesPerSample != 1 && bytesPerSample != 2) {
    throw std::runtime_error("Invalid byesPerSample, only 1 or 2 are valid.");
  }
  WavHdr hdr;
  strncpy(reinterpret_cast<char*>(hdr.RIFF), "RIFF", 4);
  strncpy(reinterpret_cast<char*>(hdr.WAVE), "WAVE", 4);

  strncpy(reinterpret_cast<char*>(hdr.fmt), "fmt ", 4);
  hdr.Subchunk1Size = 16;
  hdr.SamplesPerSec = sampleRate;
  hdr.AudioFormat = 1; // PCM
  hdr.NumOfChan = 1; // Mono
  hdr.bitsPerSample = 8*bytesPerSample;
  hdr.bytesPerSec = sampleRate * hdr.NumOfChan * hdr.bitsPerSample/8;
  hdr.blockAlign = hdr.NumOfChan * hdr.bitsPerSample/8;
  
  strncpy(reinterpret_cast<char*>(hdr.Subchunk2ID), "data", 4);
  hdr.Subchunk2Size = hdr.bitsPerSample/8 * hdr.NumOfChan * samples.size();

  hdr.ChunkSize = 32+hdr.Subchunk2Size;

  file.write(reinterpret_cast<char*>(&hdr), sizeof(WavHdr));
  for (int i = 0; i < samples.size(); i++) {
    if (bytesPerSample == 1) {
      double pcm = (samples[i]+1.0)*(1<<7);
      uint8_t sample;
      if (pcm < 0) {
        sample = 0;
      } else if (pcm > std::numeric_limits<uint8_t>::max()) {
        sample = std::numeric_limits<uint8_t>::max();
      } else {
        sample = static_cast<uint8_t>(pcm);
      }
      file.write(reinterpret_cast<char*>(&sample), sizeof(uint8_t));
    } else {
      double pcm = samples[i]*(1<<15);
      int16_t sample;
      if (pcm < std::numeric_limits<int16_t>::min()) {
        sample = std::numeric_limits<int16_t>::min();
      } else if (pcm > std::numeric_limits<int16_t>::max()) {
        sample = std::numeric_limits<int16_t>::max();
      } else {
        sample = static_cast<int16_t>(pcm);
      }
      file.write(reinterpret_cast<char*>(&sample), sizeof(int16_t));
    }
  }
}

namespace internal {

std::vector<double> GetFrequencyDomain(uint32_t sampleRate) {
  const double pi = std::acos(-1);
  // sampleRate = samples / second
  // From A0 = 440/(2^4) = 27.5 1/s
  // To C8 = A0 * 2^(7+3/12) = A0 * 2^(87/12) ~ 4186 1/s
  // y = sin(f_d*i)
  // y = sin(2*pi*f_t*t)
  // t = i / sampleRate
  // y = sin(2*pi*f_t*i/sampleRate)
  // f_d = 2*pi*f_t/sampleRate
  std::vector<double> frequencies;
  frequencies.reserve(88);
  for (int i = 0; i <= 87; i++) {
    double f_t = 27.5*std::pow(2, i/12.0);
    double f_d = 2*pi*f_t/sampleRate;
    frequencies.push_back(f_d);
  }
  return frequencies;
}

}  // namespace internal

}  // namespace audio
