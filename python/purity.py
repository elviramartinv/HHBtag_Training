import ROOT
import json
import matplotlib.pyplot as plt
import numpy as np
import numba

from statsmodels.stats.proportion import proportion_confint

from Apply import ApplyTraining

# ggf and VBF res
mass = [200]

path = "/afs/cern.ch/user/e/emartinv/public/cms-hh-bbtautau/Framework/ZZ_training_ntuples/gg_X_ZZbbtautau_M"

taggers = ["HHBtagScore", "btagDeepFlavB"]
results = {}

training_variables = '../config/training_variables.json'
params_json = '../config/params_optimized_training.json'

applier = ApplyTraining(params_json, '../config/mean_std_red.json', '../config/min_max_red.json', training_variables)
parities = [0, 1]

@numba.jit(nopython=True)
def count_matched(sorted_idx, genMatched):
    cnt = 0
    N = genMatched.shape[0]
    for n in range(N):
        if genMatched[n, sorted_idx[n, 0]] == 1 and genMatched[n, sorted_idx[n, 1]] == 1:
            cnt += 1
    return cnt

# file = path + str(mass) + ".root"
file = "/afs/cern.ch/user/e/emartinv/public/cms-hh-bbtautau/Framework/ZZ_training_ntuples/gg_X_ZZbbtautau_M200.root"
file_results = {}
error = {}
for parity in parities: 
    # results_hhbtag[parity] = {}
    # error_results_hhbtag[parity] = {}
    weights_HH = f'test_24Jan_par{abs(parity-1)}/model'
    weights_ZZ = f'ZZ_300_par{abs(parity-1)}/model'

    newHH_pred_HH, genMatched_HH = applier.apply(file, weights_HH, parity)
    newHH_sorted_HH = np.argsort(-newHH_pred_HH)
    newHH_nMatched_HH = count_matched(newHH_sorted_HH, genMatched_HH)
    purity_HH = float(newHH_nMatched_HH) / newHH_pred_HH.shape[0]
    file_results['newHHBtag'] = purity_HH

    newZZ_pred_ZZ, genMatched_ZZ = applier.apply(file, weights_ZZ, parity)
    newZZ_sorted_ZZ = np.argsort(-newZZ_pred_ZZ)
    newZZ_nMatched_ZZ = count_matched(newZZ_sorted_ZZ, genMatched_ZZ)
    purity_ZZ = float(newZZ_nMatched_ZZ) / newZZ_pred_ZZ.shape[0]
    file_results['newZZBtag'] = purity_ZZ

    lower_newHH, upper_newHH = proportion_confint(newHH_nMatched_HH, newHH_pred_HH.shape[0], alpha=0.68, method='beta')
    file_results['newHHBtag_err'] = (purity_HH - lower_newHH, upper_newHH - purity_HH)

    lower_newZZ, upper_newZZ = proportion_confint(newZZ_nMatched_ZZ, newZZ_pred_ZZ.shape[0], alpha=0.68, method='beta')
    file_results['newZZBtag_err'] = (purity_ZZ - lower_newZZ, upper_newZZ - purity_ZZ)
    # print(f'{mass} newHH_pred.shape={newHH_pred.shape} newHH_nMatched={newHH_nMatched} purity={purity}')

    df = ROOT.RDataFrame("Event", file)
    df = df.Filter("RecoJet_pt.size()>=2").Filter(f'event % 2 == {parity}')    

    num_evt = df.Count()
    df = df.Define('RecoJet_idx', 'CreateIndexes(RecoJet_pt.size())')
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
        # print(f'CI (lower): {file_results[tagger]["ci"][0]}')
        # print(f'CI (upper): {file_results[tagger]["ci"][1]}') 
        # print(f'err (lower): {file_results[tagger]["err"][0]}')
        # print(f'err (upper): {file_results[tagger]["err"][1]}')        
    results[mass] = file_results

with open("/output/ZZ/json/res_M300.json", "w") as json_file:
    json.dump(results, json_file)


# Plotting
# deepFlav_purities = [result["btagDeepFlavB"]["purity"] for result in results.values()]
# deepFlav_err = [result["btagDeepFlavB"]["err"] for result in results.values()]
# HHBtag_purities = [result["HHBtagScore"]["purity"] for result in results.values()]
# HHBtag_err = [result["HHBtagScore"]["err"] for result in results.values()]
# newHHbtag_purities = [result["newHHBtag"] for result in results.values()]
# newHHbtag_err = [result["newHHBtag_err"] for result in results.values()]
# newZZbtag_purities = [result["newZZBtag"] for result in results.values()]   
# newZZbtag_err = [result["newZZBtag_err"] for result in results.values()]

# xtick_locations = np.arange(len(masses))

# fig = plt.figure(figsize=(10, 6))
# ay = plt.gca()
# ay.errorbar(xtick_locations, deepFlav_purities, yerr=np.array(deepFlav_err).T, fmt='o', color='orange', label='DeepFlav')
# ay.errorbar(xtick_locations, HHBtag_purities, yerr=np.array(HHBtag_err).T, fmt='o', color='red', label='HHBtag v1')
# ay.errorbar(xtick_locations, newHHbtag_purities, yerr=np.array(newHHbtag_err).T, fmt='o', color='black', label='HHBtag v2')
# ay.errorbar(xtick_locations, newZZbtag_purities, yerr=np.array(newZZbtag_err).T, fmt='o', color='blue', label='ZZBtag')

# plt.xticks(xtick_locations, masses, rotation=45)

# plt.xlabel('mass [GeV]')
# plt.ylabel('Purity')
# plt.title('Res M-300')
# plt.legend()
# # plt.grid()
# plt.tight_layout()

# plt.savefig('../output/purity/purity_ggF_spin2.pdf')


#plt.show()





