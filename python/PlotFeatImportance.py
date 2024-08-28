# Create feature importance plots from PermutationFeatureImporance.py output
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import json
import os


var_par = ['channelId', 'sample_year', 'rel_met_pt_htt_pt', 'htt_scalar_pt', 'htt_eta', 
'htt_met_dphi', 'htt_pt', 'rel_jet_M_pt', 'jet_htt_deta',
'jet_eta', 'jet_pt', 'jet_htt_dphi',  'rel_jet_E_pt', 'jet_deepFlavour']

var_mapping = {
    'channelId': 'channelId',
    'sample_year': 'sample_year',
    'rel_met_pt_htt_pt': 'rel_met_pt_htt_pt',
    'htt_scalar_pt': 'htt_scalar_pt',
    'htt_eta': 'htt_eta',
    'htt_met_dphi': 'htt_met_dphi',
    'htt_pt': 'htt_pt',
    'rel_jet_M_pt': 'rel_jet_{}_M_pt',
    'jet_htt_deta': 'jet_{}_htt_deta',
    'jet_eta': 'jet_{}_eta',
    'jet_pt': 'jet_{}_pt',
    'jet_htt_dphi': 'jet_{}_htt_dphi',
    'rel_jet_E_pt': 'rel_jet_{}_E_pt',
    'jet_deepFlavour': 'jet_{}_deepFlavour'
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Name of the model without parity", required=True)
args = parser.parse_args()

feat_imp_even_path = os.path.join(args.model + "_par0", "feat_imp.json")
feat_imp_odd_path = os.path.join(args.model + "_par1", "feat_imp.json")

if not os.path.exists(feat_imp_even_path):
    raise FileNotFoundError(f"The file {feat_imp_even_path} doesn't exist.")
if not os.path.exists(feat_imp_odd_path):
    raise FileNotFoundError(f"The file {feat_imp_odd_path} doesn't exist.")

with open(feat_imp_even_path, 'r') as f:
    feat_imp_even = json.load(f)

with open(feat_imp_odd_path, 'r') as f:
    feat_imp_odd = json.load(f)

var_score_par0 = []
var_score_par1 = []

for var in var_par:
    json_var_name = var_mapping.get(var, None)
    
    if json_var_name:
        var_score_par0.append(feat_imp_even.get(json_var_name, 0.0))
    else:
        var_score_par0.append(0.0)
    
    if json_var_name:
        var_score_par1.append(feat_imp_odd.get(json_var_name, 0.0))
    else:
        var_score_par1.append(0.0)


ind = np.arange(len(var_score_par0))
fig, ax = plt.subplots()

width = 0.35 # the width of the bars
p1 = ax.barh(ind + width/2, var_score_par0, width)

p2 = ax.barh(ind - width/2, var_score_par1, width, color='mediumvioletred')

ax.set_title("Importance via feature permutation")
ax.set_ylabel('Feature')

ax.legend((p1[0], p2[0]), ('train with parity even', 'train with parity odd'))
ax.set_yticks(range(len(var_score_par1)))
ax.set_yticklabels(('channelId', 'sample_year', 'rel_met_pt_htt_pt', 'htt_scalar_pt', 'htt_eta', 
'htt_met_dphi', 'htt_pt', 'rel_jet_M_pt', 'jet_htt_deta', 
'jet_eta', 'jet_pt', 'jet_htt_dphi', 'rel_jet_E_pt', 'jet_deepFlavour'))
#ax.set_yticklabels(('rel_met_pt_htt_pt', 'htt_scalar_pt', 'htt_eta', 'htt_met_dphi', 'htt_pt', 'jet_htt_deta', 'rel_jet_M_pt', 'rel_jet_E_pt',  'jet_pt',
     #  'jet_htt_dphi', 'jet_eta', 'jet_deepFlavour'))
plt.xlabel("Importance via feature permutation")

ax.autoscale_view()
plt.savefig("output/feat_importance_" + args.model + "_GGF_2022.png", dpi=300, bbox_inches='tight')

#for only one training
# plt.barh(var_par, var_score_par0, color='teal')
# plt.ylabel("Feature")
# plt.xlabel("Importance via feature permutation")
# plt.grid(True)
# plt.draw()
# plt.savefig("feature_importance_v2.pdf",  dpi=300, bbox_inches='tight')

# plt.close()