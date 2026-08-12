#include "model.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <set>

using namespace std;
namespace fs = std::filesystem;

std::vector<Transition> load_transitions_from_folder(const std::string &folder) {
    std::vector<Transition> transitions;
    fs::path p(folder);
    if (!fs::exists(p)) throw std::runtime_error("Folder does not exist: " + folder);

    std::vector<fs::path> csv_files;
    for (const auto &entry : fs::directory_iterator(p)) {
        if (entry.is_regular_file() && entry.path().extension() == ".csv")
            csv_files.push_back(entry.path());
    }
    std::sort(csv_files.begin(), csv_files.end());

    for (const auto &file : csv_files) {
        std::ifstream in(file);
        if (!in) continue;

        std::string line;
        // read header
        if (!std::getline(in, line)) continue;
        std::vector<std::string> header;
        {
            std::istringstream hs(line);
            std::string h;
            while (std::getline(hs, h, ',')) header.push_back(h);
        }

        // find indices
        int idx_ss=-1, idx_mm=-1, idx_gt=-1;
        for (size_t i=0;i<header.size();++i) {
            if (header[i]=="SS") idx_ss=i;
            if (header[i]=="MM") idx_mm=i;
            if (header[i]=="GT") idx_gt=i;
        }
        if (idx_ss<0 || idx_mm<0 || idx_gt<0) throw std::runtime_error("CSV missing required columns in " + file.string());

        std::vector<std::string> ss;
        std::vector<float> ms, gts;

        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream lin(line);
            std::string cell;
            std::vector<std::string> cells;
            while (std::getline(lin, cell, ',')) cells.push_back(cell);
            if ((int)cells.size() <= max({idx_ss, idx_mm, idx_gt})) continue;
            ss.push_back(cells[idx_ss]);
            ms.push_back(std::stof(cells[idx_mm]));
            gts.push_back(std::stof(cells[idx_gt]));
        }

        string current="";
        vector<string> sequences;
        for (auto &s: ss){ current += s; sequences.push_back(current); }

        for (size_t i=1;i<sequences.size();++i){
            Transition t;
            t.prev_m = ms[i-1];
            t.prev_gt = gts[i-1];
            t.curr_m = ms[i];
            t.curr_gt = gts[i];
            t.delta_mass = ms[i] - ms[i-1];
            t.delta_gt = gts[i] - gts[i-1];
            t.label = sequences[i].empty() ? string("") : string(1, sequences[i].back());
            transitions.push_back(t);
        }
    }

    if (transitions.empty()) throw std::runtime_error("No transitions found in folder: " + folder);
    return transitions;
}

std::tuple<torch::Tensor, torch::Tensor, std::map<std::string,int>, std::map<int,std::string>> build_features_and_labels(const std::vector<Transition>& transitions){
    std::set<std::string> labels_set;
    for (auto &t: transitions) labels_set.insert(t.label);
    std::vector<std::string> unique_labels(labels_set.begin(), labels_set.end());
    std::map<std::string,int> label_to_id;
    std::map<int,std::string> id_to_label;
    for (size_t i=0;i<unique_labels.size();++i){ label_to_id[unique_labels[i]] = (int)i; id_to_label[(int)i] = unique_labels[i]; }

    std::vector<float> Xdata;
    std::vector<int64_t> Ydata;
    Xdata.reserve(transitions.size()*6);
    Ydata.reserve(transitions.size());

    for (auto &t: transitions){
        Xdata.push_back(t.prev_m);
        Xdata.push_back(t.prev_gt);
        Xdata.push_back(t.curr_m);
        Xdata.push_back(t.curr_gt);
        Xdata.push_back(t.delta_mass);
        Xdata.push_back(t.delta_gt);
        Ydata.push_back(label_to_id[t.label]);
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor X = torch::from_blob(Xdata.data(), {(int)transitions.size(), 6}, options).clone();
    torch::Tensor y = torch::from_blob(Ydata.data(), {(int)transitions.size()}, torch::TensorOptions().dtype(torch::kInt64)).clone();

    return {X, y, label_to_id, id_to_label};
}

MetricAutoencoderImpl::MetricAutoencoderImpl(int input_dim, int hidden_dim, int latent_dim){
    int h2 = std::max(hidden_dim/2, 4);
    encoder = torch::nn::Sequential(
        torch::nn::Linear(input_dim, hidden_dim), torch::nn::Functional(torch::relu),
        torch::nn::Linear(hidden_dim, h2), torch::nn::Functional(torch::relu),
        torch::nn::Linear(h2, latent_dim)
    );
    decoder = torch::nn::Sequential(
        torch::nn::Linear(latent_dim, h2), torch::nn::Functional(torch::relu),
        torch::nn::Linear(h2, hidden_dim), torch::nn::Functional(torch::relu),
        torch::nn::Linear(hidden_dim, input_dim)
    );
    register_module("encoder", encoder);
    register_module("decoder", decoder);
}

std::pair<torch::Tensor, torch::Tensor> MetricAutoencoderImpl::forward(torch::Tensor x){
    auto z = encoder->forward(x);
    auto recon = decoder->forward(z);
    return {recon, z};
}

torch::Tensor supervised_contrastive_loss(torch::Tensor embeddings, torch::Tensor labels, double temperature){
    auto z = torch::nn::functional::normalize(
        embeddings, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    auto similarity = torch::mm(z, z.t()) / temperature;
    int64_t batch_size = embeddings.size(0);
    auto device = embeddings.device();
    auto self_mask = torch::eye(batch_size, torch::TensorOptions().dtype(torch::kBool)).to(device);

    auto label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1);
    auto positive_mask = label_matrix & (~self_mask);

    auto logits = similarity - std::get<0>(similarity.max(1, true)).detach();
    auto exp_logits = torch::exp(logits) * (~self_mask);
    auto log_prob = logits - torch::log(exp_logits.sum(1, true) + 1e-8);

    auto positive_count = positive_mask.sum(1);
    auto valid = positive_count > 0;
    if (valid.sum().item<int>() == 0) return torch::tensor(0.0, torch::TensorOptions().device(device).requires_grad(true));

    auto mean_log_prob_positive = ((positive_mask.to(torch::kFloat) * log_prob).sum(1) / positive_count.clamp_min(1).to(torch::kFloat));
    auto loss = -mean_log_prob_positive.index({valid}).mean();
    return loss;
}
