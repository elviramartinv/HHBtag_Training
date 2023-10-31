import ROOT
import json
import matplotlib.pyplot as plt
import numpy as np
import numba

from statsmodels.stats.proportion import proportion_confint

from Apply import ApplyTraining

# ggf and VBF res
masses = [250, 260, 270, 280, 300, 320, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 1000, 1250, 1500, 1750, 2000, 2500, 3000]

#ggf non res
# masses = ['SM', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

#VBF non res

path = "/afs/cern.ch/user/e/emartinv/public/Fork_Framework/Framework/training_skim/GluGluToBulkGravitonToHHTo2B2Tau_M-"
# path = "/afs/cern.ch/user/e/emartinv/public/Fork_Framework/Framework/training_skim/GluGluToRadionToHHTo2B2Tau_M-"
# path = "/afs/cern.ch/user/e/emartinv/public/Fork_Framework/Framework/training_skim/VBFToBulkGravitonToHHTo2B2Tau_M-"
# path = "/afs/cern.ch/user/e/emartinv/public/Fork_Framework/Framework/training_skim/VBFToRadionToHHTo2B2Tau_M-"
# path = "/afs/cern.ch/user/e/emartinv/public/Fork_Framework/Framework/training_skim/GluGluToHHTo2B2Tau_node_"

taggers = ["particleNetAK4_B", "HHBtagScore", "btagDeepFlavB"]
results = {}

weights = 'newHHmodel/model'
training_variables = '../config/training_variables.json'
params_json = '../config/params_optimized_training.json'

applier = ApplyTraining(params_json, '../config/mean_std_red.json', '../config/min_max_red.json', weights, training_variables)
parity = 0

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
    error = {}
    
    newHH_pred, genMatched = applier.apply(file, parity)
    newHH_sorted = np.argsort(-newHH_pred)
    newHH_nMatched = count_matched(newHH_sorted, genMatched)
    purity = float(newHH_nMatched) / newHH_pred.shape[0]
    file_results['newHHBtag'] = purity
    
    lower_newHH, upper_newHH = proportion_confint(newHH_nMatched, newHH_pred.shape[0], alpha=0.68, method='beta')
    file_results['newHHBtag_err'] = (purity - lower_newHH, upper_newHH - purity)

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

with open("../output/purity/purity_results.json", "w") as json_file:
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
# plt.scatter(xtick_locations, HHBtag_purities, marker='o', color='red', label='HHBtag')
ay.errorbar(xtick_locations, HHBtag_purities, yerr=np.array(HHBtag_err).T, fmt='o', color='red', label='oldHHBtag')
# plt.scatter(xtick_locations,newHHbtag_purities, marker='o', color='black', label='newHHBtag')
ay.errorbar(xtick_locations, newHHbtag_purities, yerr=np.array(newHHbtag_err).T, fmt='o', color='black', label='newHHBtag')

plt.xticks(xtick_locations, masses, rotation=45)

plt.xlabel('mass [GeV]')
plt.ylabel('Purity')
plt.title('ggF spin-2')
plt.legend()
# plt.grid()
plt.tight_layout()

plt.savefig('../output/purity/purity_ggF_spin2.pdf')


#plt.show()





