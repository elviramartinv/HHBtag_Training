import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

import argparse
import json
import numpy as np

import InputsProducer as InputsProducer
import ParametrizedModel as pm
from CalculateWeigths import CreateSampleWeigts, CrossCheckWeights

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

#model = tf.keras.models.load_model(args.weights)
model = pm.HHModel(var_pos, '../config/mean_std_red.json', '../config/min_max_red.json', params)
opt = getattr(tf.keras.optimizers, params['optimizers'])(learning_rate=10 ** params['learning_rate_exp'])
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              weighted_metrics=[pm.sel_acc_2, pm.sel_acc_3, pm.sel_acc_4])
model.build(X.shape)
model.load_weights(args.weights)

model.summary()
pred = model.call(X[0:10, :, :])
print(pred)
