#include "model.h"

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <csv_folder> [epochs] [batch_size]" << std::endl;
        return 1;
    }

    std::string folder = argv[1];
    int epochs = (argc > 2) ? std::stoi(argv[2]) : 200;
    int batch_size = (argc > 3) ? std::stoi(argv[3]) : 64;
    if (epochs < 1 || batch_size < 1) {
        std::cerr << "epochs and batch_size must both be positive" << std::endl;
        return 1;
    }

    std::vector<Transition> transitions;
    try {
        transitions = load_transitions_from_folder(folder);
    } catch (const std::exception &e) {
        std::cerr << "Error loading transitions: " << e.what() << std::endl;
        return 1;
    }

    torch::Tensor X, y;
    std::map<std::string,int> label_to_id;
    std::map<int,std::string> id_to_label;
    std::tie(X, y, label_to_id, id_to_label) = build_features_and_labels(transitions);

    // normalize
    auto mean = X.mean(0);
    auto std = X.std(/*dim=*/0, /*unbiased=*/false);
    std = torch::where(std < 1e-8, torch::ones_like(std), std);
    auto X_norm = (X - mean) / std;

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) device = torch::Device(torch::kCUDA);

    MetricAutoencoder model(/*input_dim=*/6, /*hidden_dim=*/32, /*latent_dim=*/2);
    model->to(device);

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
    torch::nn::MSELoss mse;

    model->train();
    for (int epoch=0; epoch<epochs; ++epoch){
        double epoch_loss = 0.0;
        size_t n = 0;
        auto permutation = torch::randperm(X_norm.size(0), torch::TensorOptions().dtype(torch::kInt64));
        for (int64_t start = 0; start < X_norm.size(0); start += batch_size) {
            auto count = std::min<int64_t>(batch_size, X_norm.size(0) - start);
            auto indices = permutation.narrow(0, start, count);
            auto batch_x = X_norm.index_select(0, indices).to(device);
            auto batch_y = y.index_select(0, indices).to(device);

            optimizer.zero_grad();
            auto out = model->forward(batch_x);
            auto recon = out.first;
            auto z = out.second;

            auto recon_loss = mse(recon, batch_x);
            auto metric_loss = supervised_contrastive_loss(z, batch_y, 0.1);
            auto total = recon_loss + 0.5 * metric_loss;
            total.backward();
            optimizer.step();

            epoch_loss += total.item<double>() * batch_x.size(0);
            n += batch_x.size(0);
        }
        epoch_loss /= (n==0?1:n);
        if ((epoch+1) % 10 == 0 || epoch==0) {
            std::cout << "Epoch " << (epoch+1) << "/" << epochs << " Loss=" << epoch_loss << std::endl;
        }
    }

    model->eval();
    {
        torch::NoGradGuard no_grad;
        auto embeddings = model->forward(X_norm.to(device)).second.to(torch::kCPU);
        std::cout << "Learned " << embeddings.size(0)
                  << " embeddings with latent dimension " << embeddings.size(1)
                  << std::endl;
    }

    // save model and normalization
    try {
        torch::save(model, "autoencoder_model.pt");
        // save mean and std
        std::ofstream normf("norm.txt");
        auto mean_acc = mean.to(torch::kCPU);
        auto std_acc = std.to(torch::kCPU);
        for (int i=0;i<mean_acc.size(0);++i){ normf << mean_acc[i].item<float>() << (i+1<mean_acc.size(0)?' ':'\n'); }
        for (int i=0;i<std_acc.size(0);++i){ normf << std_acc[i].item<float>() << (i+1<std_acc.size(0)?' ':'\n'); }
        normf.close();
        std::cout << "Saved model to autoencoder_model.pt and normalization to norm.txt" << std::endl;
    } catch (const std::exception &e){
        std::cerr << "Error saving model: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
