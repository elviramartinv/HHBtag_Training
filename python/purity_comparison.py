import json
import matplotlib.pyplot as plt
import numpy as np
# Carga los archivos JSON
# with open('output/Run3_purities/json/0410deepFlav.json') as f:
#     deepFlav_json = json.load(f)
# with open('output/Run3_purities/json/HHBTag_v3.1_2023BPix.json') as f:
#     PNet_json = json.load(f)
# with open('output/Run3_purities/json/HHBTag_v3.1_2023.json') as f:
#     v3_json = json.load(f)
# with open('output/Run3_purities/json/0810Robust.json') as f:
#     Robust_json = json.load(f)
# with open('output/Run3_purities/json/HHBTag_v2.0_2023BPix.json') as f:
#     v2_json = json.load(f)
with open('output/Run3_purities/json/2024_1809.json') as f:
    model2024_json = json.load(f)
with open('output/Run3_purities/json/2024_v3p2.json') as f:
    v3p2on2024_json = json.load(f)

# Extrae los datos de newHHBtag para cada año
# deepFlav = [v['btagDeepFlavB']['purity'] for k, v in PNet_json.items()]
# deepFlav_err = [v['btagDeepFlavB']['err'] for k, v in PNet_json.items()]
# PNet = [v['btagPNetB']['purity'] for k, v in PNet_json.items()]
# PNet_err = [v['btagPNetB']['err'] for k, v in PNet_json.items()]
# Robust = [v['btagRobustParTAK4B']['purity'] for k, v in PNet_json.items()]
# Robust_err = [v['btagRobustParTAK4B']['err'] for k, v in PNet_json.items()]
# # HHBtag_v2 = [v['HHBtagScore_v2']['purity'] for k, v in v2_json.items()]
# # HHBtag_v2_err = [v['HHBtagScore_v2']['err'] for k, v in v2_json.items()]
# HHBtag_v2 = [v['newHHBtag'] for k, v in v2_json.items()]
# HHBtag_v2_err = [v['newHHBtag_err'] for k, v in v2_json.items()]
# HHBtag_v3 = [v['newHHBtag'] for k, v in PNet_json.items()]
# HHBtag_v3_err = [v['newHHBtag_err'] for k, v in PNet_json.items()]
model2024 = [v['newHHBtag'] for k, v in model2024_json.items()]
model2024_err = [v['newHHBtag_err'] for k, v in model2024_json.items()]
v3p2on2024 = [v['newHHBtag'] for k, v in v3p2on2024_json.items()]
v3p2on2024_err = [v['newHHBtag_err'] for k, v in v3p2on2024_json.items()]

# x_labels = ['ggF SM', 'ggF 0-1-0', 'ggF 2.45-1-0', 'ggF 5-1-0', 'VBF SM', 'VBF 1-0-1']
x_labels = ['SM (0)', '2p45-1-0 (6)', '5-1-0 (7)', '0-1-0 (13)', '0-1-1 (14)', '1-1-0p1 (15)', '1-1-p35 (16)', '1-1-3 (17)', '1-1-m2 (18)']
points = [k for k in model2024_json.keys()]


# deepFlav_err = np.array(deepFlav_err).T
# PNet_err = np.array(PNet_err).T
# Robust_err = np.array(Robust_err).T
# HHBtag_v2_err = np.array(HHBtag_v2_err).T
# HHBtag_v3_err = np.array(HHBtag_v3_err).T
model2024_err = np.array(model2024_err).T
v3p2on2024_err = np.array(v3p2on2024_err).T

# Crea el gráfico
plt.figure(figsize=(12, 7))
# plt.errorbar(points, deepFlav, yerr=deepFlav_err, fmt='o', label='DeepFlav', color='orange')
# plt.errorbar(points, PNet, yerr=PNet_err, fmt='o', label='ParticleNet', color='green')  
# plt.errorbar(points, Robust, yerr=Robust_err, fmt='o', label='RobustPar', color='blue')
# plt.errorbar(points, HHBtag_v3, yerr=HHBtag_v3_err, fmt='o', label='HHBtag v3 Run3', color='black')
# plt.errorbar(points, HHBtag_v2, yerr=HHBtag_v2_err, fmt='o', label='HHBtag v2 Run2', color='red')
plt.errorbar(points, model2024, yerr=model2024_err, fmt='o', label='HHBtag 2024 ', color='purple')
plt.errorbar(points, v3p2on2024, yerr=v3p2on2024_err, fmt='o', label='HHBtag v3p2 ', color='green')
# plt.xticks(points, rotation=45)
plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45)
# plt.ylim(0.93, 0.985)
# Añade etiquetas y título
plt.subplots_adjust(bottom=0.35)
plt.xlabel('Signal sample', fontsize=14)
plt.ylabel('Purity', fontsize=14)
plt.title('2024')
plt.legend(fontsize=14, ncol=2)
plt.savefig('output/Run3_purities/2024_v3p2comparison.pdf')
plt.savefig('output/Run3_purities/2024_v3p2comparison.png', dpi=300)

#plt.savefig('newHHBtag.png', dpi=300)