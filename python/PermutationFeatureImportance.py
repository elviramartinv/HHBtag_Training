# Evaluate the importance of the training variables using the Permutation Feature Importance method
# This file is part of https://github.com/hh-italian-group/hh-bbtautau.

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger

import argparse
import numba
import json
import numpy as np
import pandas as pd
import os

import InputsProducer
import ParametrizedModel as pm
#import BayesianOptimizationCustom as bo
from CalculateWeigths import CreateSampleWeigts, CrossCheckWeights

import matplotlib.pyplot as plt
import json
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weights")
parser.add_argument("-p", "--parity", type=int)
parser.add_argument("-training_variables", "--training_variables")
parser.add_argument("-params_json", "--params_json")
parser.add_argument("-f", "--file", nargs='+')

args = parser.parse_args()

file_name = pm.ListToVector(args.file)
with open(args.params_json) as f:
    params = json.load(f)

data = InputsProducer.CreateRootDF(file_name, args.parity, False, True)
X, Y, Z, var_pos, var_pos_z, var_name = InputsProducer.CreateXY(data, args.training_variables)

model = pm.HHModel(var_pos, '../config/mean_std_red.json', '../config/min_max_red.json', params)
opt = getattr(tf.keras.optimizers, params['optimizers'])(learning_rate=10 ** params['learning_rate_exp']) ### AdamW
# opt = getattr(tf.keras.optimizers.legacy, params['optimizers'])(learning_rate=10 ** params['learning_rate_exp']) ### Adam
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              weighted_metrics=[pm.sel_acc_2, pm.sel_acc_3, pm.sel_acc_4])
model.build(X.shape)
#model.load_weights(args.weights, by_name=True)
model.load_weights(args.weights)

@numba.njit
def Shuffle3D (X, var_idx):
    num_valid = np.count_nonzero(X[:, :, 0 ])
    X_vector  = np.zeros(num_valid)
    n = 0
    for evt in range(X.shape[0]):
        for jet in range (X.shape[1]):
            if X[evt, jet, 0 ] > 0.5:
                X_vector[n] = X[evt, jet, var_idx]
                n += 1

    X_s = np.copy(X)
    np.random.shuffle(X_vector)

    n = 0
    for evt in range(X.shape[0]):
        for jet in range (X.shape[1]):
            if X[evt, jet, 0 ] > 0.5:
                X_s[evt, jet, var_idx] = X_vector[n]
                n += 1
    return X_s

print("shape", X.shape)

ref_score = pm.sel_acc(Y, model.predict(X, batch_size=params['batch_size']), 2, 2, True, True)

def Compare(X, var_pos, ref_score, params):
    dscore = []
    dscore_var = {}
    for var_idx in range(1, len(var_pos)):
        X_shuffled = Shuffle3D(X, var_idx)
        var_score = pm.sel_acc(Y, model.predict(X_shuffled,  batch_size=params['batch_size'], steps=None), 2, 2,True, True)
        dscore_var[var_idx] = { var_name[var_idx] : ref_score - var_score}
        dscore.append(ref_score - var_score)
        

    # dscore.sort(reverse = True)
    return dscore, dscore_var

dscore, dscore_var =  Compare(X, var_pos, ref_score, params)
print(dscore_var)
sorted_dscore_var = {k: v for k, v in sorted(dscore_var.items(), key=lambda item: list(item[1].values())[0])}
sorted_params = [list(item[1].keys())[0] for item in sorted_dscore_var.items()]
sorted_dscore_list = [list(item.values())[0] for item in sorted_dscore_var.values()]
# print("sorted_dscore", sorted_params)
# print("sorted_dscore_var", sorted_dscore_list)

# json_path = os.path.join(os.path.dirname(args.weights), '../', 'feat_imp.py')
scores_dict = {list(item.keys())[0]: list(item.values())[0] for item in sorted_dscore_var.values()}

# Ruta para guardar el archivo JSON
scores_dict = {list(item.keys())[0]: float(list(item.values())[0]) for item in sorted_dscore_var.values()}

# Ruta para guardar el archivo JSON
output_path = os.path.join(os.path.dirname(args.weights), 'feat_imp.json')

# Guardar el diccionario como archivo JSON
with open(output_path, 'w') as json_file:
    json.dump(scores_dict, json_file, indent=4)

print(f"Scores guardados en {output_path}")



# import json
# dscore_var = {k: {k2: float(v2) for k2, v2 in v.items()} for k, v in dscore_var.items()}
# with open('feat_imp.py', 'w') as f:
#     json.dump(dscore_var, f)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(Y, model.predict(X) > 0.5)
# print(cm)

df = pd.DataFrame(X.reshape(-1, X.shape[-1]), columns=var_name.values())
corr_matrix = df.corr()
print(corr_matrix)

plt.figure(figsize=(15, 14))
sns.set(font_scale=2) 
heatmap = sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap='PRGn', annot=True, linewidths=0.5, linecolor='white', square=True, annot_kws={"size": 7})

heatmap.tick_params(axis='both', which='major', labelsize=12)

plt.xticks(rotation=90)

model_path = os.path.join(os.path.dirname(args.weights))

plt.savefig(model_path + f"corr_matrix_par{args.parity}.png", dpi=300)

