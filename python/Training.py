# Definition of the training for the HH-btag NN
# This file is part of https://github.com/hh-italian-group/hh-bbtautau.

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
#print(gpus)
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger

import argparse
import json
import numpy as np

import InputsProducer as InputsProducer
import ParametrizedModel as pm
from CalculateWeigths import CreateSampleWeigts, CrossCheckWeights
import ROOT

from keras.callbacks import Callback

parser = argparse.ArgumentParser()
parser.add_argument("-params", "--params")
parser.add_argument("-output", "--output", required=True)
parser.add_argument("-training_variables", "--training_variables")
parser.add_argument("-n_epoch", "--n_epoch", type=int)
parser.add_argument("-patience", "--patience", type=int)
parser.add_argument("-validation_split", "--validation_split", type=float)
parser.add_argument("-parity", "--parity", type=int)
parser.add_argument("-f", "--file", nargs='+')
parser.add_argument('-seed', '--seed', nargs='?', default=12345, type=int)

args = parser.parse_args()

file_name = pm.ListToVector(args.file)

with open(args.params) as f:
    params = json.load(f)

class WeightsSaver(Callback):
  def __init__(self, N):
    self.N = N
    self.epoch = 0

  def on_epoch_end(self, epoch, logs={}):
    if self.epoch % self.N == 0:
        name = '{}_par{}_weight_epoch_{}.tf'.format(args.output, args.parity, self.epoch )
        #name = '{}_par{}_weight_epoch_{}.h5'.format(args.output, args.parity, self.epoch )
        self.model.save_weights(name)
    self.epoch += 1

def PerformTraining(file_name, n_epoch, params):
    np.random.seed(args.seed)
    data = InputsProducer.CreateRootDF(file_name, 0, True, False)
    X, Y, Z, var_pos, var_pos_z, var_name = InputsProducer.CreateXY(data, args.training_variables)
    print(var_pos)
    # raise RuntimeError('stop')
    w = CreateSampleWeigts(X, Z)
    # print(f'graviton=radion check : {CrossCheckWeights(Z, X, w, True, False, False, 2018)}')
    # print(f'res=nonres check : {CrossCheckWeights(Z, X, w, False, True, False, 2018)}')
    # print(f'channel check : {CrossCheckWeights(Z, X, w, False, False, True, 2018)}')
    # raise RuntimeError('stop')
    Y = Y.reshape(Y.shape[0:2])
    tf.random.set_seed(args.seed)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(file_dir)
    cfg_dir = os.path.join(base_dir, 'config')

    model = pm.HHModel(var_pos, os.path.join(cfg_dir, 'mean_std_red.json'),
                       os.path.join(cfg_dir, 'min_max_red.json'), params)
    model.call(X[0:1,:,:])
    opt = getattr(tf.keras.optimizers, params['optimizers'])(lr=10 ** params['learning_rate_exp'])
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  weighted_metrics=[pm.sel_acc_2])

    model.build(X.shape)
    model.summary()
    #print(Y[0, :])
    #print(w)
    #raise RuntimeError('stop')

    if os.path.exists(args.output):
       raise RuntimeError(f'Output {args.output} already exists')
    os.makedirs(args.output)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_sel_acc_2', mode='max', patience=args.patience)
    csv_logger = CSVLogger(os.path.join(args.output, 'history.csv'), append=False, separator=',')

    save_best_only =  tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.output, 'model'),
                                                         monitor='val_sel_acc_2',  mode='max', save_best_only=True, save_weights_only=False, verbose=1)

    model.fit(X, Y, validation_split=args.validation_split, epochs=args.n_epoch, batch_size=params['batch_size'],
              callbacks=[csv_logger, save_best_only, early_stop],verbose=1)

    #model.save_weights('{}_par{}_final_weights.tf'.format(args.output, args.parity))
    #model.save_weights('{}_par{}_final_weights.h5'.format(args.output, args.parity))

# with open('{}_par{}_params.json'.format(args.output, args.parity), 'w') as f:
#     f.write(json.dumps(params, indent=4))

PerformTraining(file_name, args.n_epoch, params)