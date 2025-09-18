import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

import argparse
import numba
import json
import numpy as np

import InputsProducer
import ParametrizedModel as pm
from CalculateWeigths import CreateSampleWeigts, CrossCheckWeights

from sklearn.inspection import permutation_importance

class FeatureImportance:
    def __init__(self, params_json, mean_std_json, min_max_red_json, model_path, training_variables):
        with open(params_json) as f:
            self.params = json.load(f)
        self.model_built = False
        self.training_variables = training_variables
        self.model_path = model_path
        self.mean_std_json = mean_std_json
        self.min_max_red_json = min_max_red_json

    def base_model(self, file_name, parity):
        data = InputsProducer.CreateRootDF(file_name, parity, False, False)
        X, Y, Z, var_pos, var_pos_z, var_name = InputsProducer.CreateXY(data, self.training_variables)

        if not self.model_built:
            self.model = pm.HHModel(var_pos, self.mean_std_json, self.min_max_red_json, self.params)
            opt = getattr(tf.keras.optimizers, self.params['optimizers'])(learning_rate=10 ** self.params['learning_rate_exp'])
            self.model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      weighted_metrics=[pm.sel_acc_2, pm.sel_acc_3, pm.sel_acc_4])
            self.model.build(X.shape)
            self.model.load_weights(self.model_path)
            self.model.summary()
            self.model_built = True
        model = self.model()
        # perm = permutation_importance(self.model, X, Y, n_repeats=10, random_state=123)  
        return model

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights")
    parser.add_argument("-p", "--parity", type=int)
    parser.add_argument("-training_variables", "--training_variables")
    parser.add_argument("-params_json", "--params_json")
    parser.add_argument("-f", "--file", nargs='+')

    args = parser.parse_args()

    feature_importance = FeatureImportance(
    args.params_json, '../config/mean_std_red.json', '../config/min_max_red.json', args.weights, args.training_variables
    )

    result = feature_importance.base_model(args.file, args.parity)

    var_name = ['jet_pt', 'jet_eta', 'rel_jet_M_pt', 'rel_jet_E_pt', 'jet_htt_deta', 'jet_deepFlavour', 'jet_htt_dphi', 'htt_pt', 'htt_eta', 'htt_met_dphi', 'rel_met_pt_htt_pt', 'htt_scalar_pt']

    # Crear gráfico de barras
    sorted_idx = result.importances_mean.argsort()[::-1]
    plt.bar(range(len(var_name)), result.importances_mean[sorted_idx], yerr=result.importances_std[sorted_idx])
    plt.xticks(range(len(var_name)), np.array(var_name)[sorted_idx], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Permutation Importance")
    plt.show()




