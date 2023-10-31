# Create feature importance plots
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

var_par0 = ['jet_pt', 'jet_eta', 'rel_jet_M_pt', 'rel_jet_E_pt', 'jet_htt_deta', 'jet_deepFlavour', 'jet_htt_dphi', 'htt_pt', 'htt_eta', 'htt_met_dphi', 'rel_met_pt_htt_pt', 'htt_scalar_pt']
var_score_par0 = [0.05816585, 0.09522861, 0.008184969, 0.009945214, 0.004855454, 0.1095351, 0.077480555, 0.002496481, 0.0007056594, 0.0006699562, 0.0003286004, 0.0004941821]

var_par1 = ['jet_pt', 'jet_eta', 'rel_jet_M_pt', 'rel_jet_E_pt', 'jet_htt_deta', 'jet_deepFlavour', 'jet_htt_dphi', 'htt_pt', 'htt_eta', 'htt_met_dphi', 'rel_met_pt_htt_pt', 'htt_scalar_pt']
var_score_par1 = [0.057702005, 0.094718635, 0.007929444, 0.009313524, 0.004612088, 0.10868251, 0.07647717, 0.0023852587, 0.0004775524, 0.0007380843, 0.00017881393, 0.00037288666]

ind = np.arange(len(var_score_par0))
fig, ax = plt.subplots()

width = 0.35 # the width of the bars
p1 = ax.barh(ind + width/2, var_score_par0, width)

p2 = ax.barh(ind - width/2, var_score_par1, width, color='mediumvioletred')

ax.set_title("Importance via feature permutation")
ax.set_ylabel('Feature')

ax.legend((p1[0], p2[0]), ('train with parity even', 'train with parity odd'))
ax.set_yticks(range(len(var_score_par1)))
ax.set_yticklabels(('jet_pt', 'jet_eta', 'rel_jet_M_pt', 'rel_jet_E_pt', 'jet_htt_deta', 'jet_deepFlavour', 'jet_htt_dphi', 'htt_pt', 'htt_eta', 'htt_met_dphi', 'rel_met_pt_htt_pt', 'htt_scalar_pt'))
plt.xlabel("Importance via feature permutation")

ax.autoscale_view()
plt.savefig("feature_importance_both.pdf", dpi=300, bbox_inches='tight')

#for only one training
plt.barh(var_par0, var_score_par0, color='teal')
plt.ylabel("Feature")
plt.xlabel("Importance via feature permutation")
plt.grid(True)
plt.draw()
plt.savefig("feature_importance_v2.png",  dpi=300, bbox_inches='tight')

plt.close()