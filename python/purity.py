import ROOT
import json
import matplotlib.pyplot as plt
import numpy as np
import numba

from Apply import ApplyTraining

masses = [250, 500, 1000, 1500, 3000]
path = "/afs/cern.ch/user/e/emartinv/public/Fork_Framework/Framework/training_skim/GluGluToRadionToHHTo2B2Tau_M-"

taggers = ["particleNetAK4_B", "HHBtagScore", "btagDeepFlavB"]
results = {}

weights = 'ggf_model/model'
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

    print("applying training")
    newHH_pred, genMatched = applier.apply(file, parity)
    print("applied training")
    newHH_sorted = np.argsort(-newHH_pred)
    newHH_nMatched = count_matched(newHH_sorted, genMatched)
    purity = float(newHH_nMatched) / newHH_pred.shape[0]
    file_results['newHH'] = purity

    print(f'{mass} newHH_pred.shape={newHH_pred.shape} newHH_nMatched={newHH_nMatched} purity={purity}')

    df = ROOT.RDataFrame("Event", file)
    df = df.Filter("RecoJet_pt.size()>=2").Filter(f'event % 2 == {parity}')
    num_evt = df.Count()
    df = df.Define('RecoJet_idx', 'CreateIndexes(RecoJet_pt.size())')
    for tagger in taggers:
        df = df.Define(f"RecoJet_{tagger}_idx_sorted", f"ReorderObjects(RecoJet_{tagger}, RecoJet_idx)")
        df = df.Define(f"{tagger}_FirstTwoJetsMatched", f"RecoJet_idx.size() >= 2 && RecoJet_genMatched.at(RecoJet_{tagger}_idx_sorted.at(0)) == 1 && RecoJet_genMatched.at(RecoJet_{tagger}_idx_sorted.at(1)) == 1")

        num_matches = df.Filter(f"{tagger}_FirstTwoJetsMatched").Count()
        purity = float(num_matches.GetValue()) / num_evt.GetValue()
        print(f'{mass} {tagger} num_matches={num_matches.GetValue()} num_evt={num_evt.GetValue()} purity={purity}')

        file_results[tagger] = purity


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



