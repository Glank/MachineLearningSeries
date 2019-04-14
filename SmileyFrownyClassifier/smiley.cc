#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using std::string;
using std::vector;

namespace smiley {

constexpr int kWidth = 20;
constexpr int kHeight = 20;

struct Sample {
  vector<float> data;
  bool isSmiley;
};

void ParseSample(const string& line, bool isSmiley, Sample* out) {
  out->isSmiley = isSmiley;
  for (const auto& c : line) {
    if (c == '0') {
      out->data.push_back(0);
    } else if (c == '1') {
      out->data.push_back(1);
    } else {
      std::cerr << "Invalid sample char: " << c << std::endl;
      std::exit(1);
    }
  }
  if (out->data.size() != kWidth*kHeight) {
    std::cerr << "Invalid sample size: " << out->data.size() << std::endl;
    std::exit(1);
  }
}

std::ostream& operator<<(std::ostream& o, const Sample& s) {
  if (s.isSmiley) {
    o << "=)";
  } else {
    o << "=(";
  }
  for (int y = 0; y < kHeight; y++) {
    o << std::endl;
    for (int x = 0; x < kWidth; x++) {
      o << (s.data[x+kWidth*y]==1 ? '#' : ' ');
    }
  }
  return o;
}

void ReadSamples(const string& file, bool areSmiles, vector<Sample>* out) {
  string line;
  std::ifstream in(file);
  if (!in.is_open()) {
    std::cerr << "Could not open: " << file << std::endl;
    std::exit(1);
  }
  while (std::getline(in, line)) {
    Sample sample;
    ParseSample(line, areSmiles, &sample);
    out->push_back(std::move(sample));
  }
}

struct ModelState {
  float weights[kWidth*kHeight];
};

// Get the initial weights, each as an independent normal variable.
ModelState InitModel() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> randNorm(0, 1);
  
  ModelState state;
  for (int i = 0; i < kWidth*kHeight; i++) {
    state.weights[i] = randNorm(gen);
  }
  return state;
};

std::ostream& operator<<(std::ostream& o, const ModelState& s) {
  for (int y = 0; y < kHeight; y++) {
    if (y != 0) {
      o << std::endl;
    }
    for (int x = 0; x < kWidth; x++) {
      float w = s.weights[x+kWidth*y];
      if (w < -1) {
        o << '-';
      } else if (w > 1) {
        o << '+';
      } else {
        o << '0';
      }
    }
  }
  return o;
}

float Evaluate(const Sample& sample, const ModelState& s) {
  float result = 0;
  for (int i = 0; i < kWidth*kHeight; i++) {
    result += sample.data[i]*s.weights[i];
  }
  return result;
}

float Error(const Sample& sample, const ModelState& s) {
  float evaluation = Evaluate(sample, s);
  if (-1 < evaluation && evaluation < 1) {
    // The evaluation was too close to call.
    return 0.5;
  }
  if ((evaluation > 0) != sample.isSmiley) {
    // The evaluation was wrong.
    return 1;
  }
  // The evaluation was correct.
  return 0;
}

float TotalError(const vector<Sample> samples, const ModelState& s) {
  float err = 0;
  for (const Sample& sample : samples) {
    err += Error(sample, s);
  }
  return err;
}

// Mutates the current state into the given output mem.
class Mutator {
 private:
  std::random_device rd_;
  std::mt19937 gen_;
  std::normal_distribution<float> randNorm_;
  std::uniform_real_distribution<float> randUniform_;
  static constexpr float kMutationRate = 0.1;

 public:
  Mutator() :rd_(), gen_(rd_()), randNorm_(0,1), randUniform_(0,1) {}
  
  void Mutate(const ModelState& curState, ModelState* out) {
    for (int i = 0; i < kWidth*kHeight; i++) {
      if (randUniform_(gen_) < kMutationRate) {
        // 10% of the time mutate the i'th weight by some
        // standard normally distributed amount.
        out->weights[i] = curState.weights[i]+randNorm_(gen_);
      } else {
        out->weights[i] = curState.weights[i];
      }
    }
  }
};

void Shuffle(vector<Sample>* samples) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(samples->begin(), samples->end(), gen);
}

void Pop10Percent(vector<Sample>* allSamples, vector<Sample>* verificationSamples) {
  int vSamples = allSamples->size()/10;
  for (int i = 0; i < vSamples; i++) {
    verificationSamples->push_back(allSamples->back());
    allSamples->pop_back();
  }
}

// Export the weights to std::out to be used in our application.
void PrintWeights(const ModelState& s) {
  for (int i = 0; i < kWidth*kHeight; i++) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << s.weights[i];
  }
}

} // namespace smiley

int main() {
  vector<smiley::Sample> samples;
  smiley::ReadSamples("samples/smileys.txt", true, &samples);
  smiley::ReadSamples("samples/frownies.txt", false, &samples);
  smiley::Shuffle(&samples);

  // Set aside 10% of sampes to ensure that the model isn't overtraining.
  vector<smiley::Sample> verificationSamples;
  Pop10Percent(&samples, &verificationSamples);

  smiley::ModelState curState = smiley::InitModel(); 

  smiley::Mutator mut;
  smiley::ModelState newState;

  float curErr = smiley::TotalError(samples, curState);
  for (int t = 0; t < 1000; t++) {
    if (t%10 == 0) {
      std::cout << "Trial " << t << ": " << curErr << std::endl;
    }
    mut.Mutate(curState, &newState);
    float newErr = smiley::TotalError(samples, newState);
    if (newErr <= curErr) {
      curState = newState;
      curErr = newErr;
    }
  }

  std::cout << curState << std::endl;
  
  std::cout << "Final training error: " <<
    (smiley::TotalError(samples, curState)/samples.size()) << std::endl;
  std::cout << "Final verification error: " <<
    (smiley::TotalError(verificationSamples, curState)/verificationSamples.size()) << std::endl;

  std::cout << std::endl << std::endl;
  PrintWeights(curState);
  std::cout << std::endl << std::endl;
  
  return 0;
}
