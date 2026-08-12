#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include <torch/torch.h>

struct Transition {
    float prev_m, prev_gt, curr_m, curr_gt, delta_mass, delta_gt;
    std::string label;
};

std::vector<Transition> load_transitions_from_folder(const std::string &folder);
std::tuple<torch::Tensor, torch::Tensor, std::map<std::string,int>, std::map<int,std::string>> build_features_and_labels(const std::vector<Transition>& transitions);

struct MetricAutoencoderImpl : torch::nn::Module {
    MetricAutoencoderImpl(int input_dim=6, int hidden_dim=32, int latent_dim=2);
    torch::nn::Sequential encoder{nullptr}, decoder{nullptr};
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
};

TORCH_MODULE(MetricAutoencoder);

torch::Tensor supervised_contrastive_loss(torch::Tensor embeddings, torch::Tensor labels, double temperature=0.1);
