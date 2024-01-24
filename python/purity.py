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

# ggf and VBF res
#masses = [250, 260, 270, 280, 300, 320, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000, 1250, 1500, 2000, 2500, 3000]
# masses = [250, 260, 280, 300, 320, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000, 1500, 3000]
#ggf non res
masses = ['SM', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# masses = ['SM']
#masses = [250,300,500]
#VBF non res

# path = "/afs/cern.ch/user/e/emartinv/public/Ntuple_prod/training_ntuples/GluGluToBulkGravitonToHHTo2B2Tau_M-"
# path = "/afs/cern.ch/user/e/emartinv/public/Ntuple_prod/training_ntuples/GluGluToRadionToHHTo2B2Tau_M-"
# path = "/afs/cern.ch/user/e/emartinv/public/Ntuple_prod/training_ntuples/VBFToBulkGravitonToHHTo2B2Tau_M-"
# path = "/afs/cern.ch/user/e/emartinv/public/Ntuple_prod/training_ntuples/VBFToRadionToHHTo2B2Tau_M-"
path = "/afs/cern.ch/user/e/emartinv/public/training_ntuples_RunII/2018_GluGluToHHTo2B2Tau_node_" 

taggers = ["particleNetAK4_B", "HHBtagScore", "btagDeepFlavB"]
results = {}
results_hhbtag = {}
error_results_hhbtag = {}

training_variables = '../config/training_variables.json'
params_json = '../config/params_optimized_training.json'

applier = ApplyTraining(params_json, '../config/mean_std_red.json', '../config/min_max_red.json', training_variables)
parities = [0, 1]

data = {"newHH_pred": [], "cpp_scores": [], "event": []}


@numba.jit(nopython=True)
def count_matched(sorted_idx, genMatched):
    cnt = 0
    N = genMatched.shape[0]
    for n in range(N):
        if genMatched[n, sorted_idx[n, 0]] == 1 and genMatched[n, sorted_idx[n, 1]] == 1:
            cnt += 1
    return cnt

for mass in masses: 
    file = path + str(mass) + ".root"
    file_results = {}
    error_results = {}

    for parity in parities:
        
        results_hhbtag[parity] = {}
        error_results_hhbtag[parity] = {}
        weights = f'RunII_19Jan_par{abs(parity-1)}/model'
        
        newHH_pred, genMatched, events, cpp_scores = applier.apply(file, weights, parity)
        newHH_sorted = np.argsort(-newHH_pred)
        newHH_nMatched = count_matched(newHH_sorted, genMatched)
        purity = float(newHH_nMatched) / newHH_pred.shape[0]
        file_results[f'newHHBtag_{parity}'] = purity

        # data["newHH_pred"].extend(newHH_pred)
        # data["cpp_scores"].extend(cpp_scores)
        # data["event"].extend(events)
        
        lower_newHH, upper_newHH = proportion_confint(newHH_nMatched, newHH_pred.shape[0], alpha=0.68, method='beta')
        error_results[f'newHHBtag_ci_{parity}'] = (lower_newHH, upper_newHH)
        error_results[f'nMatched_{parity}'] = (newHH_nMatched)
        error_results[f'pred_{parity}'] = (newHH_pred.shape[0])
        
        #file_results[f'newHHBtag_err_{parity}'] = (purity - lower_newHH, upper_newHH - purity)
        
        #print(f'{parity} and {mass} newHH_pred.shape={newHH_pred.shape} newHH_nMatched={newHH_nMatched} purity={purity}')
        results_hhbtag[parity][mass] = file_results
        error_results_hhbtag[parity][mass] = error_results
        
    total_purity = {}
    total_err = {}
    total_purity = (
        (results_hhbtag[0][mass]['newHHBtag_0'] + results_hhbtag[1][mass]['newHHBtag_1']) / 2
    )
    lower_newHH_total, upper_newHH_total = proportion_confint(np.add(error_results_hhbtag[0][mass]['nMatched_0'], error_results_hhbtag[1][mass]['nMatched_1']),
                                           np.add(error_results_hhbtag[0][mass]['pred_0'], error_results_hhbtag[1][mass]['pred_1']),
                                           alpha = 0.68, method = 'beta')
    
    error_low = np.sqrt(lower_newHH_total**2 + (lower_newHH_total - np.minimum(error_results_hhbtag[0][mass]['newHHBtag_ci_0'][0], 
                                                                                    error_results_hhbtag[1][mass]['newHHBtag_ci_1'][0]))**2)
    
    error_up = np.sqrt(upper_newHH_total**2 + (upper_newHH_total - np.maximum(error_results_hhbtag[0][mass]['newHHBtag_ci_0'][1], 
                                                                                    error_results_hhbtag[1][mass]['newHHBtag_ci_1'][1]))**2)
    
    
    file_results['newHHBtag_err'] = (total_purity - error_low, error_up - total_purity)
    # raise RuntimeError('stop')
    file_results['newHHBtag'] = total_purity
    # print("file", file)

    # for key in data:
    #     data[key] = np.array(data[key])

    # sort_indices = np.argsort(data["event"])

    # for key in data:
    #     data[key] = data[key][sort_indices]

    # out_path = "out_scores/"
    # outFile = out_path + f'scores_{mass}.root'
    # with uproot.recreate(outFile, compression=uproot.LZMA(9)) as out_file:
    #     out_file["Events"] = data
        
        # {
        #     'HHbtag': newHH_pred,
        #     'HHbtag_cpp': cpp_scores,
        #     'event': events
        # }




    # raise RuntimeError("stop")

    df = ROOT.RDataFrame("Event", file)
    df = df.Filter("RecoJet_pt.size()>=2")   
    # df = df.Define("Htt_mass", "np.sqrt(2 * HttCandidate_leg0_pt * HttCandidate_leg1_pt * (np.cosh(HttCandidate_leg0_eta - HttCandidate_leg1_pt) - np.cos(HttCandidate_leg0_phi - HttCandidate_leg1_phi))")
    
    # df = df.Define('Hbb_mass', 'HttCandidate_leg0_pt + httCandidate_leg1_pt')
    
    # RecoJet_pt.at(RecoJet_genMatched==1

    num_evt = df.Count()
    df = df.Define('RecoJet_idx', 'CreateIndexes(RecoJet_pt.size())')
    for tagger in taggers:
        df = df.Define(f"RecoJet_{tagger}_idx_sorted", f"ReorderObjects(RecoJet_{tagger}, RecoJet_idx)")
        df = df.Define(f"{tagger}_FirstTwoJetsMatched", f"RecoJet_idx.size() >= 2 && RecoJet_genMatched.at(RecoJet_{tagger}_idx_sorted.at(0)) == 1 && RecoJet_genMatched.at(RecoJet_{tagger}_idx_sorted.at(1)) == 1")

        num_matches = df.Filter(f"{tagger}_FirstTwoJetsMatched").Count()
        purity = float(num_matches.GetValue()) / num_evt.GetValue()
        #print(f'{mass} {tagger} num_matches={num_matches.GetValue()} num_evt={num_evt.GetValue()} purity={purity}')

        # Calculate ci
        lower, upper = proportion_confint(num_matches.GetValue(), num_evt.GetValue(), alpha=0.68, method='beta')
        
        file_results[tagger] = {
            'purity': purity,
            'ci': (lower, upper),
            'err': (purity - lower, upper-purity)
        }      
    results[mass] = file_results
    #print("results", results[mass])

with open("output/RunII_19Jan/json/2018_nonres.json", "w") as json_file:
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
ay.errorbar(xtick_locations, HHBtag_purities, yerr=np.array(HHBtag_err).T, fmt='o', color='red', label='oldHHBtag')
#ay.scatter(xtick_locations,newHHbtag_purities, marker='o', color='black', label='newHHBtag')
ay.errorbar(xtick_locations, newHHbtag_purities, yerr=np.array(newHHbtag_err).T, fmt='o', color='black', label='newHHBtag')

plt.xticks(xtick_locations, masses, rotation=45)

plt.xlabel('EFT benchmark')
plt.ylabel('Purity')
plt.title('ggF nonRes')
plt.legend()
# plt.grid()
plt.tight_layout()

plt.savefig('output/RunII_19Jan/ggf_nonRes_2018.pdf')
#plt.savefig('purity_ggF_nonres.png', dpi=300)


#plt.show()





