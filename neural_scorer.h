#ifndef NEURAL_SCORER_H_
#define NEURAL_SCORER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ctcdecode/ctcdecode/src/path_trie.h"
#include "ctcdecode/ctcdecode/src/lm_scorer.h"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

class Neural_Scorer:public Scorer {
public:
  Neural_Scorer(double alpha,
         double beta,
         const std::string &lm_path,
         const std::vector<std::string> &vocabulary,
         int max_order,
         const std::string& neural_lm_path,
         bool have_dictionary);
  ~Neural_Scorer();

  double get_log_cond_prob(const std::vector<std::string> &words) override;

  double get_sent_log_prob(const std::vector<std::string> &words) override;

  // return the max order
  size_t get_max_order() override { return max_order_; }

  // return the dictionary size of language model
  size_t get_dict_size() override { return dict_size_; }

  // retrun true if the language model is character based
  bool is_character_based() override { return is_character_based_; }

  // reset params alpha & beta
  void reset_params(float alpha, float beta) override;

  // make ngram for a given prefix
  std::vector<std::string> make_ngram(PathTrie *prefix) override;

  // trransform the labels in index to the vector of words (word based lm) or
  // the vector of characters (character based lm)
  std::vector<std::string> split_labels(const std::vector<int> &labels) override;

  void* paddle_get_scorer(double alpha,
                        double beta,
                        const char* lm_path,
                        std::vector<std::string> new_vocab,
                        int max_order,
                        const char* vocab_path,
                        bool have_dictionary);

  void paddle_release_scorer(void* scorer);

protected:
  // necessary setup: load language model, set char map, fill FST's dictionary
  void setup(const std::string &vocab_path,
             const std::vector<std::string> &vocab_list) override;

  // load language model from given path
  void load_lm(const std::string &lm_path) override;

  // fill dictionary for FST
  void fill_dictionary(bool add_space) override;

  // set char map
  void set_char_map(const std::vector<std::string> &char_list) override;

  double get_log_prob(const std::vector<std::string> &words) override;

  // translate the vector in index to string
  std::string vec2str(const std::vector<int> &input) override;

private:
  int vocabSize_;
  std::string lm_path_;
  bool have_dictionary_;
  Ort::Session* session;
  std::vector<const char*> input_node_names;
  std::vector<const char*> output_node_names;
};

#endif  // NEURAL_SCORER_H_
