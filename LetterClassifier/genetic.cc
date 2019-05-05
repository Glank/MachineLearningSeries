#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using std::string;
using std::vector;

namespace genetic {

constexpr int kWidth = 20;
constexpr int kHeight = 20;
constexpr int kNumLetters = 26;

struct Sample {
  vector<float> data;
  char letter;
};

void ParseSample(const string& line, char letter, Sample* out) {
  out->letter = letter;
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
  o << "Letter: " << s.letter;
  for (int y = 0; y < kHeight; y++) {
    o << std::endl;
    for (int x = 0; x < kWidth; x++) {
      o << (s.data[x+kWidth*y]==1 ? '#' : ' ');
    }
  }
  return o;
}

void ReadSamples(const string& file, char letter, vector<Sample>* out) {
  string line;
  std::ifstream in(file);
  if (!in.is_open()) {
    std::cerr << "Could not open: " << file << std::endl;
    std::exit(1);
  }
  while (std::getline(in, line)) {
    Sample sample;
    ParseSample(line, letter, &sample);
    out->push_back(std::move(sample));
  }
}

struct ModelState {
  float weights[kWidth*kHeight*kNumLetters];
};

// Get the initial weights, each as an independent normal variable.
ModelState InitModel() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> randNorm(0, 1);
  
  ModelState state;
  for (int i = 0; i < kWidth*kHeight*kNumLetters; i++) {
    state.weights[i] = randNorm(gen);
  }
  return state;
};

void Evaluate(const Sample& sample, const ModelState& s, float* result) {
  for (int l = 0; l < kNumLetters; l++) {
    result[l] = 0;
    for (int i = 0; i < kWidth*kHeight; i++) {
      result[l] += sample.data[i]*s.weights[l*kWidth*kHeight+i];
    }
  }
}

float Error(const Sample& sample, const ModelState& s) {
  float letters[kNumLetters];
  Evaluate(sample, s, letters);
  int expected = (int)(sample.letter-'A');
  float error = 0;
  for(int l = 0; l < kNumLetters; l++) {
    if (-1 < letters[l] && letters[l] < 1) {
      error += 0.5;
    } else if ((letters[l] > 0) != (l == expected)) {
      if (l == expected) {
        error += 10;
      } else {
        error += 1;
      }
    }
  }
  error /= kNumLetters;
  return error;
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
    for (int i = 0; i < kWidth*kHeight*kNumLetters; i++) {
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

// Export the weights to a file to be used in our application.
void SaveWeights(const ModelState& s) {
  std::ofstream out("weights.js");
  if (!out.is_open()) {
    std::cout << "Error opening 'weights.js'" << std::endl;
    std::exit(1);
  }
  out << "var weights = [";
  for (int i = 0; i < kWidth*kHeight*kNumLetters; i++) {
    if (i != 0) {
      out << ", ";
    }
    out << s.weights[i];
  }
  out << "]\n";
}

} // namespace genetic

int main() {
  vector<genetic::Sample> samples;
  for (int l = 0; l < genetic::kNumLetters; l++) {
    std::stringstream fn;
    fn << "samples/" << (char)('a'+l) << "_samples.txt";
    genetic::ReadSamples(fn.str(), (char)('A'+l), &samples);
  }
  genetic::Shuffle(&samples);

  // Set aside 10% of sampes to ensure that the model isn't overtraining.
  vector<genetic::Sample> verificationSamples;
  Pop10Percent(&samples, &verificationSamples);

  genetic::ModelState curState = genetic::InitModel(); 

  genetic::Mutator mut;
  genetic::ModelState newState;

  float curErr = genetic::TotalError(samples, curState);
  for (int t = 0; t < 10000; t++) {
    std::cout << "Trial " << t << ": " << curErr << std::endl;
    mut.Mutate(curState, &newState);
    float newErr = genetic::TotalError(samples, newState);
    if (newErr <= curErr) {
      curState = newState;
      curErr = newErr;
    }
  }

  std::cout << "Final training error: " <<
    (genetic::TotalError(samples, curState)/samples.size()) <<
    std::endl;
  std::cout << "Final verification error: " <<
    (genetic::TotalError(verificationSamples, curState)/verificationSamples.size()) <<
    std::endl;

  SaveWeights(curState);
  
  return 0;
}
