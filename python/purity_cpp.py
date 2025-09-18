import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import ROOT
import json
import matplotlib.pyplot as plt
import numpy as np
import numba
import uproot

from statsmodels.stats.proportion import proportion_confint

from Apply import ApplyTraining
import InputsProducer as InputsProducer


# ggf and VBF res
masses = [250, 260, 270, 280, 300, 320, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000, 1250, 1500, 2000, 2500, 3000]
# masses = [250, 260, 280, 300, 320, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000, 1500, 3000]
#ggf non res
# masses = ['SM', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# masses = ['SM', 1]
#masses = [250,300,500]
#VBF non res

path = "/afs/cern.ch/user/e/emartinv/public/training_ntuples_RunII/2018_GluGluToBulkGravitonToHHTo2B2Tau_M-"
path_new = "/afs/cern.ch/user/e/emartinv/public/Ntuple_prod/newHH_ntuples/GluGluToBulkGravitonToHHTo2B2Tau_M-"
# path = "/afs/cern.ch/user/e/emartinv/public/training_ntuples_RunII/2018_GluGluToRadionToHHTo2B2Tau_M-"
# path_new = "/afs/cern.ch/user/e/emartinv/public/Ntuple_prod/newHH_ntuples/GluGluToRadionToHHTo2B2Tau_M-"
# path = "/afs/cern.ch/user/e/emartinv/public/Ntuple_prod/training_ntuples/VBFToBulkGravitonToHHTo2B2Tau_M-"
# path_new = "/afs/cern.ch/user/e/emartinv/public/Ntuple_prod/newHH_ntuples/VBFToBulkGravitonToHHTo2B2Tau_M-"
# path = "/afs/cern.ch/user/e/emartinv/public/Ntuple_prod/training_ntuples/VBFToRadionToHHTo2B2Tau_M-"
# path_new = "/afs/cern.ch/user/e/emartinv/public/Ntuple_prod/newHH_ntuples/VBFToRadionToHHTo2B2Tau_M-"
# path = "/afs/cern.ch/user/e/emartinv/public/training_ntuples_RunII/2018_GluGluToHHTo2B2Tau_node_" 
# path_new = "/afs/cern.ch/user/e/emartinv/public/Ntuple_prod/newHH_ntuples/GluGluToHHTo2B2Tau_node_"



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

taggers = ["particleNetAK4_B", "HHBtagScore", "btagDeepFlavB"]
results = {}
results_hhbtag = {}
error_results_hhbtag = {}


for mass in masses: 
    file = path + str(mass) + ".root"
    file_new = path_new + str(mass) + ".root"
    file_results = {}
    error_results = {}

    df = ROOT.RDataFrame("Event", file)
    df = df.Filter("RecoJet_pt.size()>=2")   
    df_new = ROOT.RDataFrame("Event", file_new)
    df_new = df_new.Filter("RecoJet_pt.size()>=2")
    # df = df.Define("Htt_mass", "np.sqrt(2 * HttCandidate_leg0_pt * HttCandidate_leg1_pt * (np.cosh(HttCandidate_leg0_eta - HttCandidate_leg1_pt) - np.cos(HttCandidate_leg0_phi - HttCandidate_leg1_phi))")
    
    # df = df.Define('Hbb_mass', 'HttCandidate_leg0_pt + httCandidate_leg1_pt')
    
    # RecoJet_pt.at(RecoJet_genMatched==1

    num_evt = df.Count()
    num_evt_new = df_new.Count()
    df = df.Define('RecoJet_idx', 'CreateIndexes(RecoJet_pt.size())')
    df_new = df_new.Define('RecoJet_idx', 'CreateIndexes(RecoJet_pt.size())')

    df_new = df_new.Define("RecoJet_HHBtagScore_idx_sorted", "ReorderObjects(RecoJet_HHBtagScore, RecoJet_idx)")
    df_new = df_new.Define("HHBtagScore_FirstTwoJetsMatched", "RecoJet_idx.size() >= 2 && RecoJet_genMatched.at(RecoJet_HHBtagScore_idx_sorted.at(0)) == 1 && RecoJet_genMatched.at(RecoJet_HHBtagScore_idx_sorted.at(1)) == 1")
    
    num_matches_new = df_new.Filter("HHBtagScore_FirstTwoJetsMatched").Count()
   
    purity = float(num_matches_new.GetValue()) / num_evt_new.GetValue()
    file_results['newHHBtag'] = purity
    # print(f'{mass} newHH', purity)

    lower_newHH, upper_newHH = proportion_confint(num_matches_new.GetValue(), num_evt_new.GetValue(), alpha=0.68, method='beta')
    file_results['newHHBtag_err'] = (purity - lower_newHH, upper_newHH - purity)

    for tagger in taggers:
        df = df.Define(f"RecoJet_{tagger}_idx_sorted", f"ReorderObjects(RecoJet_{tagger}, RecoJet_idx)")
        df = df.Define(f"{tagger}_FirstTwoJetsMatched", f"RecoJet_idx.size() >= 2 && RecoJet_genMatched.at(RecoJet_{tagger}_idx_sorted.at(0)) == 1 && RecoJet_genMatched.at(RecoJet_{tagger}_idx_sorted.at(1)) == 1")

        num_matches = df.Filter(f"{tagger}_FirstTwoJetsMatched").Count()
        purity = float(num_matches.GetValue()) / num_evt.GetValue()
        # print(f'{mass} {tagger} num_matches={num_matches.GetValue()} num_evt={num_evt.GetValue()} purity={purity}')

        # Calculate ci
        lower, upper = proportion_confint(num_matches.GetValue(), num_evt.GetValue(), alpha=0.68, method='beta')
        
        file_results[tagger] = {
            'purity': purity,
            'ci': (lower, upper),
            'err': (purity - lower, upper-purity)
        }      
    results[mass] = file_results
    #print("results", results[mass])

with open("HHBtag_14dec_performance/purity_results_ggF_spin2.json", "w") as json_file:
    json.dump(results, json_file)


particleNet_purities = [result["particleNetAK4_B"]["purity"] for result in results.values()]
particleNet_err = [result["particleNetAK4_B"]["err"] for result in results.values()]
deepFlav_purities = [result["btagDeepFlavB"]["purity"] for result in results.values()]
deepFlav_err = [result["btagDeepFlavB"]["err"] for result in results.values()]
HHBtag_purities = [result["HHBtagScore"]["purity"] for result in results.values()]
HHBtag_err = [result["HHBtagScore"]["err"] for result in results.values()]
newHHbtag_purities = [result["newHHBtag"] for result in results.values()]
newHHbtag_err = [result["newHHBtag_err"] for result in results.values()]

xtick_locations = np.arange(len(masses))

fig = plt.figure(figsize=(10, 6))
ay = plt.gca()

# plt.scatter(xtick_locations, particleNet_purities, marker='o', color='green', label='ParticleNet')
ay.errorbar(xtick_locations, particleNet_purities, yerr=np.array(particleNet_err).T, fmt='o', color='green', label='ParticleNet')
# plt.scatter(xtick_locations, deepFlav_purities, marker='o', color='orange', label='DeepFlav')
ay.errorbar(xtick_locations, deepFlav_purities, yerr=np.array(deepFlav_err).T, fmt='o', color='orange', label='DeepFlav')
#plt.scatter(xtick_locations, HHBtag_purities, marker='o', color='red', label='HHBtag')
ay.errorbar(xtick_locations, HHBtag_purities, yerr=np.array(HHBtag_err).T, fmt='o', color='red', label='HH-BTag v1')
#ay.scatter(xtick_locations,newHHbtag_purities, marker='o', color='black', label='newHHBtag')
ay.errorbar(xtick_locations, newHHbtag_purities, yerr=np.array(newHHbtag_err).T, fmt='o', color='black', label='HH-BTag v2')

plt.xticks(xtick_locations, masses, rotation=45)

plt.xlabel('mass [GeV]')
plt.ylabel('Purity')
# plt.title('ggHH non resonant')
plt.legend()
# plt.grid()
plt.tight_layout()

# plt.savefig('HHBtag_14dec_performance/purity_ggF_spin2.pdf', dpi=300)
plt.savefig('HHBtag_14dec_performance/purity_ggF_spin2.png', dpi=300)


#plt.show()





