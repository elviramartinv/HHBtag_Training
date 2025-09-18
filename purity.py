import ROOT
import json
import matplotlib.pyplot as plt
import numpy as np

from Apply import ApplyTraining

masses = [250, 500, 1000, 1500, 3000]
path = "/afs/cern.ch/user/e/emartinv/public/Fork_Framework/Framework/training_skim/GluGluToRadionToHHTo2B2Tau_M-"

ROOT.gInterpreter.Declare('''
using LorentzVectorXYZ = ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>;
using LorentzVectorM = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;
using LorentzVectorE = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double>>;
using RVecI = ROOT::VecOps::RVec<int>;
using RVecS = ROOT::VecOps::RVec<size_t>;
using RVecUC = ROOT::VecOps::RVec<UChar_t>;
using RVecF = ROOT::VecOps::RVec<float>;
using RVecB = ROOT::VecOps::RVec<bool>;
using RVecVecI = ROOT::VecOps::RVec<RVecI>;
using RVecLV = ROOT::VecOps::RVec<LorentzVectorM>;
using RVecSetInt = ROOT::VecOps::RVec<std::set<int>>;

RVecS CreateIndexes(size_t vecSize){
  RVecS i(vecSize);
  std::iota(i.begin(), i.end(), 0);
  return i;
}

template<typename V>
RVecI ReorderObjects(const V& varToOrder, const RVecI& indices, size_t nMax=std::numeric_limits<size_t>::max())
{
  RVecI ordered_indices = indices;
  std::sort(ordered_indices.begin(), ordered_indices.end(), [&](int a, int b) {
    return varToOrder.at(a) > varToOrder.at(b);
  });
  const size_t n = std::min(ordered_indices.size(), nMax);
  ordered_indices.resize(n);
  return ordered_indices;
}
''')

#ROOT.gInterpreter.Declare('''
##include <ROOT/RVec.hxx>
#
#bool CheckFirstTwoJets(const ROOT::VecOps::RVec<float>& v, const ROOT::VecOps::RVec<int>& genMatched) {
#    if (v.size() < 2) return false;
#    return (genMatched[v[0]] == 1 && genMatched[v[1]] == 1);
#}
#''')

taggers = ["particleNetAK4_B", "HHBtagScore", "btagDeepFlavB"]
results = {}

weights = 'ggf_model/model'
training_variables = '../config/training_variables.json'
params_json = '../config/params_optimized_training.json'

applier = ApplyTraining(params_json, '../config/mean_std_red.json', '../config/min_max_red.json', weights, training_variables)

for mass in masses:
    file = path + str(mass) + ".root"
    df = ROOT.RDataFrame("Event", file)
    file_results = {}
    parity = 0
    df = df.Filter("RecoJet_pt.size()>=2").Filter(f'event % 2 == {parity}')
    num_evt = df.Count()
    df = df.Define('RecoJet_idx', 'CreateIndexes(RecoJet_pt.size())')
    for tagger in taggers:
        df = df.Define(f"RecoJet_{tagger}_idx_sorted", f"ReorderObjects(RecoJet_{tagger}, RecoJet_idx)")
        df = df.Define(f"{tagger}_FirstTwoJetsMatched", f"RecoJet_idx.size() >= 2 && RecoJet_genMatched.at(RecoJet_{tagger}_idx_sorted.at(0)) == 1 && RecoJet_genMatched.at(RecoJet_{tagger}_idx_sorted.at(1)) == 1")

        num_matches = df.Filter(f"{tagger}_FirstTwoJetsMatched").Count()
        purity = float(num_matches.GetValue()) / num_evt.GetValue()

        file_results[tagger] = purity

    newHH_pred, genMatched = applier.apply(file, parity)
    newHH_sorted = np.argsort(-newHH_pred)
    newHH_matched = (genMatched[:, newHH_sorted[:, 0]] == 1) & (genMatched[:, newHH_sorted[:, 1]] == 1)
    newHH_nMatched = np.count_nonzero(newHH_matched)
    purity = float(newHH_nMatched) / num_evt.GetValue()
    file_results['newHH'] = purity

    results[mass] = file_results

with open("purity_results.json", "w") as json_file:
    json.dump(results, json_file)

# with open("purity_results.json", "r") as json_file:
#     results = json.load(json_file)


particleNet_purities = [result["particleNetAK4_B"] for result in results.values()]
deepFlav_purities = [result["btagDeepFlavB"] for result in results.values()]
HHBtag_purities = [result["HHBtagScore"] for result in results.values()]
newHHbtag_purities = [result["newHH"] for result in results.values()]

plt.figure(figsize=(10, 6))
plt.plot(masses, particleNet_purities, marker='o', color='green', label='ParticleNet')
plt.plot(masses, deepFlav_purities, marker='o', color='orange', label='DeepFlav')
plt.plot(masses, HHBtag_purities, marker='o', color='red', label='HHBtag')
plt.plot(masses,newHHbtag_purities, marker='o', color='black', label='newHH')
plt.xlabel('Mass')
plt.ylabel('Purity')
plt.title('Purity vs Mass for Different Taggers')
plt.legend()
plt.grid()


plt.savefig('purity_plot.pdf')


#plt.show()



