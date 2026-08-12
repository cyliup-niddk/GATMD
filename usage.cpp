#include "model.h"
#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>

struct Peak { double mass, rt; };
struct Stats { double mass_mean, mass_std, rt_mean, rt_std; };
struct Score { double combined, latent, mass, rt, delta_mass, delta_rt; };
struct Edge { int from=-1, to=-1; std::string monomer; Score score{}; };

static std::pair<torch::Tensor,torch::Tensor> load_norm(const std::string& path) {
    std::ifstream in(path); if (!in) throw std::runtime_error("Cannot open normalization file: " + path);
    std::vector<float> v(12); for (auto& x:v) if (!(in>>x)) throw std::runtime_error("norm.txt must contain 12 numbers");
    auto mean=torch::from_blob(v.data(),{6},torch::kFloat32).clone();
    auto std=torch::from_blob(v.data()+6,{6},torch::kFloat32).clone(); return {mean,std};
}
static std::vector<Peak> load_peaks(const std::string& path) {
    std::ifstream in(path); if(!in) throw std::runtime_error("Cannot open peaks CSV: "+path);
    std::string line; std::getline(in,line); std::vector<Peak> out;
    while(std::getline(in,line)){ if(line.empty()) continue; std::istringstream s(line); std::string a,b;
      if(std::getline(s,a,',')&&std::getline(s,b,',')) out.push_back({std::stod(a),std::stod(b)}); }
    if(out.empty()) throw std::runtime_error("No peaks in: "+path); return out;
}
static double gaussian(double x,double mean,double sigma){ double z=(x-mean)/sigma; return std::exp(-0.5*z*z); }

int main(int argc,char** argv){
  if(argc<5){ std::cerr<<"Usage: "<<argv[0]<<" <model.pt> <norm.txt> <training_csv_folder> <peaks.csv> [initial_monomer]\n"; return 1; }
  try {
    torch::manual_seed(42); torch::Device device(torch::cuda::is_available()?torch::kCUDA:torch::kCPU);
    MetricAutoencoder model(6,32,2); torch::load(model,argv[1]); model->to(device); model->eval();
    auto [mean,stddev]=load_norm(argv[2]); mean=mean.to(device); stddev=stddev.to(device);
    auto transitions=load_transitions_from_folder(argv[3]);
    torch::Tensor X,y; std::map<std::string,int> label_to_id; std::map<int,std::string> id_to_label;
    std::tie(X,y,label_to_id,id_to_label)=build_features_and_labels(transitions);
    std::map<int,torch::Tensor> centroids;
    { torch::NoGradGuard guard; auto z=torch::nn::functional::normalize(model->encoder->forward((X.to(device)-mean)/stddev),torch::nn::functional::NormalizeFuncOptions().dim(1));
      auto yd=y.to(device); for(auto const& kv:id_to_label){ auto c=z.index({yd==kv.first}).mean(0).unsqueeze(0); centroids[kv.first]=torch::nn::functional::normalize(c,torch::nn::functional::NormalizeFuncOptions().dim(1)).squeeze(0); }}
    std::map<std::string,Stats> stats;
    for(auto const& kv:label_to_id){ std::vector<double> dm,dr; for(auto const&t:transitions) if(t.label==kv.first){dm.push_back(t.delta_mass);dr.push_back(t.delta_gt);}
      auto calc=[](const std::vector<double>&v){double m=0;for(double x:v)m+=x;m/=v.size();double q=0;for(double x:v)q+=(x-m)*(x-m);return std::pair<double,double>{m,std::sqrt(q/v.size())};};
      auto [mm,ms]=calc(dm);auto [rm,rs]=calc(dr);stats[kv.first]={mm,std::max(ms,0.5),rm,std::max(rs,0.1)}; }
    auto peaks=load_peaks(argv[4]); std::sort(peaks.begin(),peaks.end(),[](auto&a,auto&b){return a.mass<b.mass;});
    auto classify=[&](const Peak&a,const Peak&b){
      float raw[6]={(float)a.mass,(float)a.rt,(float)b.mass,(float)b.rt,(float)(b.mass-a.mass),(float)(b.rt-a.rt)};
      auto f=torch::from_blob(raw,{1,6},torch::kFloat32).clone().to(device); torch::Tensor emb;
      {torch::NoGradGuard guard;emb=torch::nn::functional::normalize(model->encoder->forward((f-mean)/stddev),torch::nn::functional::NormalizeFuncOptions().dim(1)).squeeze(0);}
      std::pair<std::string,Score> best; best.second.combined=-1;
      for(auto const& kv:label_to_id){double cosine=torch::cosine_similarity(emb.unsqueeze(0),centroids[kv.second].unsqueeze(0)).item<double>();auto st=stats[kv.first];
        Score s; s.delta_mass=b.mass-a.mass;s.delta_rt=b.rt-a.rt;s.latent=(cosine+1)/2;s.mass=gaussian(s.delta_mass,st.mass_mean,st.mass_std);s.rt=gaussian(s.delta_rt,st.rt_mean,st.rt_std);s.combined=.35*s.latent+.50*s.mass+.15*s.rt;
        if(s.combined>best.second.combined)best={kv.first,s}; }
      return best; };
    const double min_edge=.55,reward=.40; int n=peaks.size(); std::vector<double> best(n,0);std::vector<int> prev(n,-1),length(n,1);std::vector<Edge> edges(n);
    for(int j=0;j<n;++j)for(int i=0;i<j;++i){if(peaks[j].rt<=peaks[i].rt)continue;auto [mon,s]=classify(peaks[i],peaks[j]);if(s.combined<min_edge)continue;double candidate=best[i]+std::log(s.combined+1e-8)+reward;if(candidate>best[j]){best[j]=candidate;prev[j]=i;length[j]=length[i]+1;edges[j]={i,j,mon,s};}}
    int end=0;for(int i=1;i<n;++i)if(std::pair{length[i],best[i]}>std::pair{length[end],best[end]})end=i;
    std::vector<int> nodes;std::vector<Edge> path;for(int cur=end;cur!=-1;cur=prev[cur]){nodes.push_back(cur);if(prev[cur]!=-1)path.push_back(edges[cur]);}std::reverse(nodes.begin(),nodes.end());std::reverse(path.begin(),path.end());
    std::string sequence=argc>5?argv[5]:"";for(auto&e:path)sequence+=e.monomer;
    std::cout<<std::fixed<<std::setprecision(3)<<"Predicted sequence: "<<sequence<<"\nSelected peptide peaks:\n";for(int i:nodes)std::cout<<"mass="<<peaks[i].mass<<", RT="<<peaks[i].rt<<"\n";
    std::cout<<"Transition details:\n";for(auto&e:path)std::cout<<e.from<<" -> "<<e.to<<" monomer="<<e.monomer<<" combined="<<e.score.combined<<" latent="<<e.score.latent<<" mass="<<e.score.mass<<" rt="<<e.score.rt<<"\n";
    std::cout<<"Path score: "<<best[end]<<"\n"; return 0;
  } catch(const std::exception&e){std::cerr<<"Error: "<<e.what()<<"\n";return 1;}
}
