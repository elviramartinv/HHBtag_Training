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
parser.add_argument('--training_variables', '-t', type=str, required=True, help='Path to training_variables json')
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
    #         '2022_VBFHHto2B2Tau_CV_m1p83_C2V_3p57_kl_m3p39']
    masses = ['2022_VBFHHto2B2Tau_SM', '2022_VBFHHto2B2Tau_1',
            '2022_VBFHHto2B2Tau_2', '2022_VBFHHto2B2Tau_3',
            '2022_VBFHHto2B2Tau_4', '2022_VBFHHto2B2Tau_5',
            '2022_VBFHHto2B2Tau_6', '2022_VBFHHto2B2Tau_7',
            '2022_VBFHHto2B2Tau_8']
elif args.year == "2022EE":
    # masses = ['2022EE_VBFHHto2B2Tau_CV_1_C2V_0_kl_1', '2022EE_VBFHHto2B2Tau_CV_1_C2V_1_kl_1',
    #         '2022EE_VBFHHto2B2Tau_CV_1p74_C2V_1p37_kl_14p4', '2022EE_VBFHHto2B2Tau_CV_m0p012_C2V_0p030_kl_10p2',
    #         '2022EE_VBFHHto2B2Tau_CV_m0p758_C2V_1p44_kl_m19p3', '2022EE_VBFHHto2B2Tau_CV_m0p962_C2V_0p959_kl_m1p43',
    #         '2022EE_VBFHHto2B2Tau_CV_m1p21_C2V_1p94_kl_m0p94', '2022EE_VBFHHto2B2Tau_CV_m1p60_C2V_2p72_kl_m1p36',
    #         '2022EE_VBFHHto2B2Tau_CV_m1p83_C2V_3p57_kl_m3p39']
    masses = ['2022EE_VBFHHto2B2Tau_SM', '2022EE_VBFHHto2B2Tau_1',
            '2022EE_VBFHHto2B2Tau_2', '2022EE_VBFHHto2B2Tau_3',
            '2022EE_VBFHHto2B2Tau_4', '2022EE_VBFHHto2B2Tau_5',
            '2022EE_VBFHHto2B2Tau_6', '2022EE_VBFHHto2B2Tau_7',
            '2022EE_VBFHHto2B2Tau_8']

elif args.year == "2023":
    # masses = ['2023_VBFHHto2B2Tau_CV_1_C2V_0_kl_1', '2023_VBFHHto2B2Tau_CV_1_C2V_1_kl_1',
    #       '2023_VBFHHto2B2Tau_CV_1p74_C2V_1p37_kl_14p4', '2023_VBFHHto2B2Tau_CV_m0p012_C2V_0p030_kl_10p2',
    #       '2023_VBFHHto2B2Tau_CV_m0p758_C2V_1p44_kl_m19p3', '2023_VBFHHto2B2Tau_CV_m0p962_C2V_0p959_kl_m1p43',
    #       '2023_VBFHHto2B2Tau_CV_m1p21_C2V_1p94_kl_m0p94', '2023_VBFHHto2B2Tau_CV_m1p60_C2V_2p72_kl_m1p36',
    #       '2023_VBFHHto2B2Tau_CV_m1p83_C2V_3p57_kl_m3p39']
    masses = ['2023_VBFHHto2B2Tau_SM', '2023_VBFHHto2B2Tau_1',
            '2023_VBFHHto2B2Tau_2', '2023_VBFHHto2B2Tau_3',
            '2023_VBFHHto2B2Tau_4', '2023_VBFHHto2B2Tau_5',
            '2023_VBFHHto2B2Tau_6', '2023_VBFHHto2B2Tau_7',
            '2023_VBFHHto2B2Tau_8']


elif args.year == "2023BPix":
    # masses = ['2023BPix_VBFHHto2B2Tau_CV_1_C2V_0_kl_1', '2023BPix_VBFHHto2B2Tau_CV_1_C2V_1_kl_1',
    #       '2023BPix_VBFHHto2B2Tau_CV_1p74_C2V_1p37_kl_14p4', '2023BPix_VBFHHto2B2Tau_CV_m0p012_C2V_0p030_kl_10p2',
    #       '2023BPix_VBFHHto2B2Tau_CV_m0p758_C2V_1p44_kl_m19p3', '2023BPix_VBFHHto2B2Tau_CV_m0p962_C2V_0p959_kl_m1p43',
    #       '2023BPix_VBFHHto2B2Tau_CV_m1p21_C2V_1p94_kl_m0p94', '2023BPix_VBFHHto2B2Tau_CV_m1p60_C2V_2p72_kl_m1p36',
    #       '2023BPix_VBFHHto2B2Tau_CV_m1p83_C2V_3p57_kl_m3p39']
    masses = ['2023BPix_VBFHHto2B2Tau_SM', '2023BPix_VBFHHto2B2Tau_1',
            '2023BPix_VBFHHto2B2Tau_2', '2023BPix_VBFHHto2B2Tau_3',
            '2023BPix_VBFHHto2B2Tau_4', '2023BPix_VBFHHto2B2Tau_5',
            '2023BPix_VBFHHto2B2Tau_6', '2023BPix_VBFHHto2B2Tau_7',
            '2023BPix_VBFHHto2B2Tau_8']



# masses = ['2023BPix_VBFHHto2B2Tau_CV_1_C2V_1_kl_1']


# path = "/eos/user/e/emartinv/vbfjets_Training/training_skims/"
path = "/eos/user/e/emartinv/vbfjets_Training/training_skims_1310_all_VBFvars_v3/"
# path = "/eos/user/e/emartinv/vbfjets_Training/gen_studies/"

results = {}
results_vbftag = {}
error_results_vbftag = {}

training_variables = args.training_variables
params_json = '../config/params_optimized_training.json'

applier = ApplyTraining(params_json, '../config/mean_std_red_VBF_2j.json', '../config/min_max_red_vbf_v2.json', training_variables)
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
        # df = df.Filter("hasVBFAK4")
        
        # Apply parity filter to match what the classifier does
        if parity >= 0 and parity <= 1:
            df = df.Filter(f'event % 2 == {parity}')
            print(f"  Applied parity filter: event % 2 == {parity}")

        # Calculate purity for old selection (isCCLUBvbfjet)
        df = df.Define('RecoJet_idx', 'CreateIndexes_0(RecoJet_pt.size())')
        
        # Filter jets that are marked as VBF jets by the old tagger
        df = df.Define("vbfjet_indices", "ROOT::VecOps::Nonzero(RecoJet_isCCLUBvbfjet)")
        
        df = df.Define("num_vbf_jets", "vbfjet_indices.size()")
        df = df.Filter("num_vbf_jets == 2")
        df = df.Define("old_vbf_matched", "vbfjet_indices.size() >= 2 && RecoJet_vbfgenMatched.at(vbfjet_indices.at(0)) == 1 && RecoJet_vbfgenMatched.at(vbfjet_indices.at(1)) == 1")
        
        # Debug prints for old selection
        
        num_vbf_jets_old = df.Filter("num_vbf_jets >= 2").Count().GetValue()
        num_old_vbf_matched = df.Filter("old_vbf_matched").Count().GetValue()
        
        # Let's check how many VBF jets we have in total and per event
        total_vbf_jets = df.Define("total_vbf_count", "static_cast<int>(ROOT::VecOps::Sum(RecoJet_isCCLUBvbfjet))").Take["int"]("total_vbf_count").GetValue()
        vbf_indices_sizes = df.Define("vbf_size", "static_cast<int>(vbfjet_indices.size())").Take["int"]("vbf_size").GetValue()
        
        # Get event numbers where old selection doesn't find 2 VBF jets
        events_no_2_vbf = df.Define("has_less_than_2_vbf", "vbfjet_indices.size() < 2").Filter("has_less_than_2_vbf")
        event_nums_no_2_vbf = events_no_2_vbf.Take["ULong64_t"]("event").GetValue()
        
        # Get event numbers where old selection finds 2 VBF jets but they don't match gen truth
        events_2_vbf_no_match = df.Define("has_2_vbf_no_match", "vbfjet_indices.size() >= 2 && !(RecoJet_vbfgenMatched.at(vbfjet_indices.at(0)) == 1 && RecoJet_vbfgenMatched.at(vbfjet_indices.at(1)) == 1)").Filter("has_2_vbf_no_match")
        event_nums_2_vbf_no_match = events_2_vbf_no_match.Take["ULong64_t"]("event").GetValue()
        
        # Debug: check the distribution of VBF jets per event
        import numpy as np
        vbf_counts = np.array(list(total_vbf_jets))
        vbf_sizes = np.array(list(vbf_indices_sizes))
        
        print(f"  VBF jets per event statistics:")
        print(f"    Min VBF jets: {np.min(vbf_counts)}, Max: {np.max(vbf_counts)}")
        print(f"    Events with 0 VBF jets: {np.sum(vbf_counts == 0)}")
        print(f"    Events with 1 VBF jet: {np.sum(vbf_counts == 1)}")
        print(f"    Events with 2+ VBF jets: {np.sum(vbf_counts >= 2)}")
        print(f"    Nonzero indices sizes - Min: {np.min(vbf_sizes)}, Max: {np.max(vbf_sizes)}")
        print(f"    Events where Nonzero gives <2 indices: {np.sum(vbf_sizes < 2)}")
        
        print(f"DEBUG {mass} parity {parity} - Old selection:")
        print(f"  Events with >=2 VBF jets (isCCLUBvbfjet): {num_vbf_jets_old}")
        print(f"  Events where both VBF jets match gen truth: {num_old_vbf_matched}")
        print(f"  Events with <2 VBF jets: {len(event_nums_no_2_vbf)}")
        if len(event_nums_no_2_vbf) > 0:
            print(f"    First 10 event numbers: {list(event_nums_no_2_vbf)[:10]}")
        print(f"  Events with 2 VBF jets but no gen match: {len(event_nums_2_vbf_no_match)}")
        if len(event_nums_2_vbf_no_match) > 0:
            print(f"    First 10 event numbers: {list(event_nums_2_vbf_no_match)[:10]}")
        # Calculate purity for old selection
        total_events = df.Count().GetValue()
        old_purity = float(num_old_vbf_matched) / num_vbf_jets_old if num_vbf_jets_old > 0 else 0.0
        
        print(f"  Total events: {total_events}")
        print(f"  Events excluded from old selection: {total_events - num_vbf_jets_old} ({100*(total_events - num_vbf_jets_old)/total_events:.1f}%)")
        print(f"  Old purity: {old_purity:.6f}")

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
        
        print(f"DEBUG {mass} parity {parity} - VBF tagger:")
        print(f"  Events analyzed by classifier: {newHH_pred.shape[0]}")
        print(f"  Events where top-2 jets match gen truth: {newHH_nMatched}")
        print(f"  VBF tagger purity: {purity:.6f}")
        print(f"  Difference in analyzed events: {total_events - newHH_pred.shape[0]}")
        print("---")

        file_results[f'vbftag_{parity}'] = purity
        file_results[f'old_selection_{parity}'] = old_purity

       
        lower_newHH, upper_newHH = proportion_confint(newHH_nMatched, newHH_pred.shape[0], alpha=0.68, method='beta')
        lower_old, upper_old = proportion_confint(num_old_vbf_matched, num_vbf_jets_old, alpha=0.68, method='beta') if num_vbf_jets_old > 0 else (0.0, 0.0)
        
        error_results[f'vbftag_ci_{parity}'] = (lower_newHH, upper_newHH)
        error_results[f'old_selection_ci_{parity}'] = (lower_old, upper_old)
        error_results[f'nMatched_{parity}'] = (newHH_nMatched)
        error_results[f'nMatched_old_{parity}'] = (num_old_vbf_matched)
        error_results[f'pred_{parity}'] = (newHH_pred.shape[0])
        error_results[f'total_events_{parity}'] = (num_vbf_jets_old)
        
        #file_results[f'vbftag_err_{parity}'] = (purity - lower_newHH, upper_newHH - purity)
        
        #print(f'{parity} and {mass} newHH_pred.shape={newHH_pred.shape} newHH_nMatched={newHH_nMatched} purity={purity}')
        results_vbftag[parity][mass] = file_results
        error_results_vbftag[parity][mass] = error_results

    all_event_numbers = np.concatenate(all_event_numbers)
    all_scores = np.concatenate(all_scores)
    all_genmatched = np.concatenate(all_genmatched) 
    
    
    total_purity = {}
    total_err = {}
    
    # Calculate combined purity for new VBF tagger
    total_purity_vbf = (
        (results_vbftag[0][mass]['vbftag_0'] + results_vbftag[1][mass]['vbftag_1']) / 2
    )
    lower_newHH_total, upper_newHH_total = proportion_confint(np.add(error_results_vbftag[0][mass]['nMatched_0'], error_results_vbftag[1][mass]['nMatched_1']),
                                           np.add(error_results_vbftag[0][mass]['pred_0'], error_results_vbftag[1][mass]['pred_1']),
                                           alpha = 0.68, method = 'beta')
    
    error_low_vbf = np.sqrt(lower_newHH_total**2 + (lower_newHH_total - np.minimum(error_results_vbftag[0][mass]['vbftag_ci_0'][0], 
                                                                                    error_results_vbftag[1][mass]['vbftag_ci_1'][0]))**2)
    
    error_up_vbf = np.sqrt(upper_newHH_total**2 + (upper_newHH_total - np.maximum(error_results_vbftag[0][mass]['vbftag_ci_0'][1], 
                                                                                    error_results_vbftag[1][mass]['vbftag_ci_1'][1]))**2)
    
    # Calculate combined purity for old selection
    total_purity_old = (
        (results_vbftag[0][mass]['old_selection_0'] + results_vbftag[1][mass]['old_selection_1']) / 2
    )
    
    print(f"\nSUMMARY for {mass}:")
    print(f"  VBF tagger purity: {total_purity_vbf:.6f}")
    print(f"  Old selection purity: {total_purity_old:.6f}")
    print(f"  Difference: {total_purity_vbf - total_purity_old:.6f}")
    print("="*50)
    lower_old_total, upper_old_total = proportion_confint(np.add(error_results_vbftag[0][mass]['nMatched_old_0'], error_results_vbftag[1][mass]['nMatched_old_1']),
                                           np.add(error_results_vbftag[0][mass]['total_events_0'], error_results_vbftag[1][mass]['total_events_1']),
                                           alpha = 0.68, method = 'beta')
    
    error_low_old = np.sqrt(lower_old_total**2 + (lower_old_total - np.minimum(error_results_vbftag[0][mass]['old_selection_ci_0'][0], 
                                                                                    error_results_vbftag[1][mass]['old_selection_ci_1'][0]))**2)
    
    error_up_old = np.sqrt(upper_old_total**2 + (upper_old_total - np.maximum(error_results_vbftag[0][mass]['old_selection_ci_0'][1], 
                                                                                    error_results_vbftag[1][mass]['old_selection_ci_1'][1]))**2)
    
    file_results['vbftag_err'] = (total_purity_vbf - error_low_vbf, error_up_vbf - total_purity_vbf)
    file_results['vbftag'] = total_purity_vbf
    file_results['old_selection_err'] = (total_purity_old - error_low_old, error_up_old - total_purity_old)
    file_results['old_selection'] = total_purity_old

    results[mass] = file_results    


with open(f"output/vbf_purities/json/{args.model}_{args.year}.json", "w") as json_file:
    json.dump(results, json_file)

xtick_locations = np.arange(len(masses))

# Map sample names to coupling labels
coupling_map = {
    'SM': '1-1-1',
    '1': '1-0-1', 
    '2': '1p74-1p37-14p4',
    '3': 'm0p012-0p030-10p2',
    '4': 'm0p758-1p44-m19p3',
    '5': 'm0p962-0p959-m1p43',
    '6': 'm1p21-1p94-m0p94',
    '7': 'm1p60-2p72-m1p36',
    '8': 'm1p83-3p57-m3p39'
}

xtick_names = []
for mass in masses:
    sample_key = mass.replace(f'{args.year}_VBFHHto2B2Tau_', '')
    xtick_names.append(coupling_map.get(sample_key, sample_key))

vbftag_purities = [result["vbftag"] for result in results.values()]
vbftag_err = [result["vbftag_err"] for result in results.values()]

old_selection_purities = [result["old_selection"] for result in results.values()]
old_selection_err = [result["old_selection_err"] for result in results.values()]

fig = plt.figure(figsize=(10, 6))
ay = plt.gca()

ay.errorbar(xtick_locations, vbftag_purities, yerr=np.array(vbftag_err).T, fmt='o', color='black', label='vbftag')
ay.errorbar(xtick_locations, old_selection_purities, yerr=np.array(old_selection_err).T, fmt='s', color='red', label='Run2 kinematic selection')

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








