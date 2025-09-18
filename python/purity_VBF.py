import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import ROOT
import json
import matplotlib.pyplot as plt
import numpy as np
import numba
import uproot

from statsmodels.stats.proportion import proportion_confint

from Apply_VBF import ApplyTraining

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--year', '-y', type=str, required=True, choices=['2022', '2022EE', '2023', '2023BPix'], help='Year to use for predictions')
parser.add_argument('--model', '-m', type=str, required=True, help='Model name to use for predictions')
args = parser.parse_args()



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

RVecS CreateIndexes_0(size_t vecSize){
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

#VBF non res
if args.year == "2022":
    # masses = ['2022_VBFHHto2B2Tau_CV_1_C2V_0_kl_1', '2022_VBFHHto2B2Tau_CV_1_C2V_1_kl_1',
    #         '2022_VBFHHto2B2Tau_CV_1p74_C2V_1p37_kl_14p4', '2022_VBFHHto2B2Tau_CV_m0p012_C2V_0p030_kl_10p2',
    #         '2022_VBFHHto2B2Tau_CV_m0p758_C2V_1p44_kl_m19p3', '2022_VBFHHto2B2Tau_CV_m0p962_C2V_0p959_kl_m1p43',
    #         '2022_VBFHHto2B2Tau_CV_m1p21_C2V_1p94_kl_m0p94', '2022_VBFHHto2B2Tau_CV_m1p60_C2V_2p72_kl_m1p36',
    #         '2022_VBFHHto2B2Tau_CV_m1p83_C2V_3p57_kl_m3p39', '2022_VBFHHto2B2Tau_CV_m2p12_C2V_3p87_kl_m5p96']
    masses = ['2022_VBFHHto2B2Tau_SM', '2022_VBFHHto2B2Tau_1',
            '2022_VBFHHto2B2Tau_2', '2022_VBFHHto2B2Tau_3',
            '2022_VBFHHto2B2Tau_4', '2022_VBFHHto2B2Tau_5',
            '2022_VBFHHto2B2Tau_6', '2022_VBFHHto2B2Tau_7',
            '2022_VBFHHto2B2Tau_8', '2022_VBFHHto2B2Tau_9']
elif args.year == "2022EE":
    masses = ['2022EE_VBFHHto2B2Tau_CV_1_C2V_0_kl_1', '2022EE_VBFHHto2B2Tau_CV_1_C2V_1_kl_1',
            '2022EE_VBFHHto2B2Tau_CV_1p74_C2V_1p37_kl_14p4', '2022EE_VBFHHto2B2Tau_CV_m0p012_C2V_0p030_kl_10p2',
            '2022EE_VBFHHto2B2Tau_CV_m0p758_C2V_1p44_kl_m19p3', '2022EE_VBFHHto2B2Tau_CV_m0p962_C2V_0p959_kl_m1p43',
            '2022EE_VBFHHto2B2Tau_CV_m1p21_C2V_1p94_kl_m0p94', '2022EE_VBFHHto2B2Tau_CV_m1p60_C2V_2p72_kl_m1p36',
            '2022EE_VBFHHto2B2Tau_CV_m1p83_C2V_3p57_kl_m3p39', '2022EE_VBFHHto2B2Tau_CV_m2p12_C2V_3p87_kl_m5p96']

elif args.year == "2023":
    masses = ['2023_VBFHHto2B2Tau_CV_1_C2V_0_kl_1', '2023_VBFHHto2B2Tau_CV_1_C2V_1_kl_1',
          '2023_VBFHHto2B2Tau_CV_1p74_C2V_1p37_kl_14p4', '2023_VBFHHto2B2Tau_CV_m0p012_C2V_0p030_kl_10p2',
          '2023_VBFHHto2B2Tau_CV_m0p758_C2V_1p44_kl_m19p3', '2023_VBFHHto2B2Tau_CV_m0p962_C2V_0p959_kl_m1p43',
          '2023_VBFHHto2B2Tau_CV_m1p21_C2V_1p94_kl_m0p94', '2023_VBFHHto2B2Tau_CV_m1p60_C2V_2p72_kl_m1p36',
          '2023_VBFHHto2B2Tau_CV_m1p83_C2V_3p57_kl_m3p39', '2023_VBFHHto2B2Tau_CV_m2p12_C2V_3p87_kl_m5p96']


elif args.year == "2023BPix":
    masses = ['2023BPix_VBFHHto2B2Tau_CV_1_C2V_0_kl_1', '2023BPix_VBFHHto2B2Tau_CV_1_C2V_1_kl_1',
          '2023BPix_VBFHHto2B2Tau_CV_1p74_C2V_1p37_kl_14p4', '2023BPix_VBFHHto2B2Tau_CV_m0p012_C2V_0p030_kl_10p2',
          '2023BPix_VBFHHto2B2Tau_CV_m0p758_C2V_1p44_kl_m19p3', '2023BPix_VBFHHto2B2Tau_CV_m0p962_C2V_0p959_kl_m1p43',
          '2023BPix_VBFHHto2B2Tau_CV_m1p21_C2V_1p94_kl_m0p94', '2023BPix_VBFHHto2B2Tau_CV_m1p60_C2V_2p72_kl_m1p36',
          '2023BPix_VBFHHto2B2Tau_CV_m1p83_C2V_3p57_kl_m3p39', '2023BPix_VBFHHto2B2Tau_CV_m2p12_C2V_3p87_kl_m5p96']



# masses = ['2023BPix_VBFHHto2B2Tau_CV_1_C2V_1_kl_1']


# path = "/eos/user/e/emartinv/vbfjets_Training/training_skims/"
path = "/eos/user/e/emartinv/vbfjets_Training/training_skims_0409/"
# path = "/eos/user/e/emartinv/vbfjets_Training/gen_studies/"

results = {}
results_vbftag = {}
error_results_vbftag = {}

training_variables = '../config/training_variables_vbf.json' 
params_json = '../config/params_optimized_training.json'

applier = ApplyTraining(params_json, '../config/mean_std_red_VBF.json', '../config/min_max_red_vbf.json', training_variables)
parities = [0, 1]

@numba.jit(nopython=True)
def count_matched(sorted_idx, vbfgenMatched):
    cnt = 0
    N = vbfgenMatched.shape[0]
    for n in range(N):
        if vbfgenMatched[n, sorted_idx[n, 0]] == 1 and vbfgenMatched[n, sorted_idx[n, 1]] == 1:
            cnt += 1
    return cnt

@numba.jit(nopython=True)
def count_matched_nobjet(sorted_idx, vbfgenMatched, bgenMatched):
    cnt = 0
    N = vbfgenMatched.shape[0]
    for n in range(N):
        if bgenMatched[n, sorted_idx[n, 0]] == 0 and bgenMatched[n, sorted_idx[n, 1]] == 0:  # Ignorar b-jets
            if vbfgenMatched[n, sorted_idx[n, 0]] == 1 and vbfgenMatched[n, sorted_idx[n, 1]] == 1:
                cnt += 1
    return cnt

for mass in masses: 
    file = path + str(mass) + ".root"
    file_results = {}
    error_results = {}
    all_event_numbers = []
    all_scores = []
    all_genmatched = []

    for parity in parities:
        
        results_vbftag[parity] = {}
        error_results_vbftag[parity] = {}
        weights = f'{args.model}_par{abs(parity-1)}/model'

        df = ROOT.RDataFrame("Event", file)

        df = df.Define('RecoJet_idx', 'CreateIndexes_0(RecoJet_pt.size())')
        df = df.Define("RecoJet_bjet_idx_sorted", "ReorderObjects(RecoJet_genMatched, RecoJet_idx)")
        df = df.Define("RecoJet_bjet_matched_vbf", "RecoJet_idx.size() >= 2 && RecoJet_vbfgenMatched.at(RecoJet_bjet_idx_sorted.at(0)) == 1 && RecoJet_vbfgenMatched.at(RecoJet_bjet_idx_sorted.at(1)) == 1")
        df = df.Define("RecoJet_bjet_matched_1vbf", "RecoJet_idx.size() >= 2 && RecoJet_vbfgenMatched.at(RecoJet_bjet_idx_sorted.at(0)) == 1")
        df = df.Define("RecoJet_bjet_matched_2vbf", "RecoJet_idx.size() >= 2 && RecoJet_vbfgenMatched.at(RecoJet_bjet_idx_sorted.at(1)) == 1")
        df = df.Define("RecoJet_bjet_matched_at_least_1vbf",
                    "RecoJet_idx.size() >= 2 && (RecoJet_vbfgenMatched.at(RecoJet_bjet_idx_sorted.at(0)) == 1 || RecoJet_vbfgenMatched.at(RecoJet_bjet_idx_sorted.at(1)) == 1)")

        num_bjets_at_least_1vbf = df.Filter("RecoJet_bjet_matched_at_least_1vbf").Count().GetValue()


        num_bjets_identified_as_vbfjets = df.Filter("RecoJet_bjet_matched_vbf").Count().GetValue()
        # print(f'Number of both b-jets identified as VBF jets in {mass} parity {parity}: {num_bjets_identified_as_vbfjets}')
        # print(f'Number of b-jet 1 identified as VBF jet in {mass} parity {parity}: {df.Filter("RecoJet_bjet_matched_1vbf").Count().GetValue()}')
        # print(f'Number of b-jet 2 identified as VBF jets in {mass} parity {parity}: {df.Filter("RecoJet_bjet_matched_2vbf").Count().GetValue()}')
        # print(f'Number of events where at least one b-jet is identified as VBF jet in {mass} parity {parity}: {num_bjets_at_least_1vbf}')
        # print("Percentage of both b-jets identified as VBF jets: ", num_bjets_identified_as_vbfjets / df.Count().GetValue())
        # print("Percentage of b-jet 1 identified as VBF jet: ", df.Filter("RecoJet_bjet_matched_1vbf").Count().GetValue() / df.Count().GetValue())
        # print("Percentage of b-jet 2 identified as VBF jets: ", df.Filter("RecoJet_bjet_matched_2vbf").Count().GetValue() / df.Count().GetValue())
        
        # print("Percentage of events where at least one b-jet is identified as VBF jet: ", num_bjets_at_least_1vbf / df.Count().GetValue())


        # df = df.Filter("RecoJet_pt.size()>=2")
        # df = df.Filter('channel == 0 || channel == 1 || channel == 2')

        newHH_pred, vbfgenMatched, bgenMatched, event_numbers = applier.apply(file, weights, parity)

        all_event_numbers.append(event_numbers)
        all_scores.append(newHH_pred)
        all_genmatched.append(vbfgenMatched)

        newHH_sorted = np.argsort(-newHH_pred)
        # print("predictions", newHH_pred)
        newHH_nMatched = count_matched(newHH_sorted, vbfgenMatched)
        # print("Number of matched VBF jets in newHH:", newHH_nMatched_2)
        # newHH_nMatched = count_matched_nobjet(newHH_sorted, vbfgenMatched, isCCLUBbjet)
        # print("Number of matched VBF jets in newHH without b-jets:", newHH_nMatched)
        purity = float(newHH_nMatched) / newHH_pred.shape[0]

        file_results[f'vbftag_{parity}'] = purity

       
        lower_newHH, upper_newHH = proportion_confint(newHH_nMatched, newHH_pred.shape[0], alpha=0.68, method='beta')
        error_results[f'vbftag_ci_{parity}'] = (lower_newHH, upper_newHH)
        error_results[f'nMatched_{parity}'] = (newHH_nMatched)
        error_results[f'pred_{parity}'] = (newHH_pred.shape[0])
        
        #file_results[f'vbftag_err_{parity}'] = (purity - lower_newHH, upper_newHH - purity)
        
        #print(f'{parity} and {mass} newHH_pred.shape={newHH_pred.shape} newHH_nMatched={newHH_nMatched} purity={purity}')
        results_vbftag[parity][mass] = file_results
        error_results_vbftag[parity][mass] = error_results

    all_event_numbers = np.concatenate(all_event_numbers)
    all_scores = np.concatenate(all_scores)
    all_genmatched = np.concatenate(all_genmatched) 
    
    
    total_purity = {}
    total_err = {}
    total_purity = (
        (results_vbftag[0][mass]['vbftag_0'] + results_vbftag[1][mass]['vbftag_1']) / 2
    )
    lower_newHH_total, upper_newHH_total = proportion_confint(np.add(error_results_vbftag[0][mass]['nMatched_0'], error_results_vbftag[1][mass]['nMatched_1']),
                                           np.add(error_results_vbftag[0][mass]['pred_0'], error_results_vbftag[1][mass]['pred_1']),
                                           alpha = 0.68, method = 'beta')
    
    error_low = np.sqrt(lower_newHH_total**2 + (lower_newHH_total - np.minimum(error_results_vbftag[0][mass]['vbftag_ci_0'][0], 
                                                                                    error_results_vbftag[1][mass]['vbftag_ci_1'][0]))**2)
    
    error_up = np.sqrt(upper_newHH_total**2 + (upper_newHH_total - np.maximum(error_results_vbftag[0][mass]['vbftag_ci_0'][1], 
                                                                                    error_results_vbftag[1][mass]['vbftag_ci_1'][1]))**2)
    
    
    file_results['vbftag_err'] = (total_purity - error_low, error_up - total_purity)
    file_results['vbftag'] = total_purity

    results[mass] = file_results    


with open(f"output/vbf_purities/json/{args.model}_{args.year}.json", "w") as json_file:
    json.dump(results, json_file)

xtick_locations = np.arange(len(masses))
xtick_names = [mass.replace(f'{args.year}_VBFHHto2B2Tau_', '') for mass in masses]

vbftag_purities = [result["vbftag"] for result in results.values()]
vbftag_err = [result["vbftag_err"] for result in results.values()]

fig = plt.figure(figsize=(10, 6))
ay = plt.gca()

ay.errorbar(xtick_locations, vbftag_purities, yerr=np.array(vbftag_err).T, fmt='o', color='black', label='vbftag')

plt.xticks(xtick_locations, xtick_names, rotation=45)

# plt.ylim(0.9500, 0.9900)

# yticks = np.arange(0.9500, 0.9900 + 0.0025, 0.0025)  
# plt.yticks(yticks)

plt.xlabel('signal sample', fontsize=14)
plt.ylabel('Purity', fontsize=14)
plt.title(f'{args.year}', fontsize=14)
plt.legend()
# plt.grid()
plt.tight_layout()

plt.savefig(f'output/vbf_purities/{args.model}_{args.year}.pdf')
plt.savefig(f'output/vbf_purities/{args.model}_{args.year}.png', dpi=300)








