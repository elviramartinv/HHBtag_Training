import os
import sys

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# if __name__ == "__main__":
#   file_dir = os.path.dirname(os.path.abspath(__file__))
#   base_dir = os.path.dirname(file_dir)
#   if base_dir not in sys.path:
#     sys.path.append(base_dir)
#   __package__ = os.path.split(file_dir)[-1]

# from .nn_deploy import save_frozen_graph

import tensorflow as tf


import argparse
import json

import InputsProducer as InputsProducer
import ParametrizedModel as pm

class ApplyTraining:
    def __init__(self, params_json, mean_std_json, min_max_red_json, training_variables):
        with open(params_json) as f:
            self.params = json.load(f)
        self.model_built = False
        self.training_variables = training_variables
        self.mean_std_json = mean_std_json
        self.min_max_red_json = min_max_red_json

    def apply(self, file_name, model_path, parity):
        data = InputsProducer.CreateRootDF(file_name, parity, False, True)
        X, Y, Z, var_pos, var_pos_z, var_name = InputsProducer.CreateXY(data, self.training_variables)
        event_sel = Z[:, 0, -2] == 15014
        X_sel = X[event_sel, :, :]
        # print('X_sel', X_sel)
        self.parity = parity
        # print('parity_apply', parity)
        self.model_path = model_path
        if not self.model_built:
            # self.model = pm.HHModel(var_pos, self.mean_std_json, self.min_max_red_json, self.params)
            self.model = tf.keras.models.load_model(model_path, custom_objects={
                'sel_acc_2': pm.sel_acc_2, 'sel_acc_3': pm.sel_acc_3, 'sel_acc_4': pm.sel_acc_4
            })
            # opt = getattr(tf.keras.optimizers, self.params['optimizers'])(learning_rate=10 ** self.params['learning_rate_exp'])
            # self.model.compile(loss='binary_crossentropy',
            #           optimizer=opt,
            #           weighted_metrics=[pm.sel_acc_2, pm.sel_acc_3, pm.sel_acc_4])
            # input_shape = (None, *X.shape[1:])
            # self.model.build(input_shape)
            # self.model.compute_output_shape(input_shape=input_shape)
            # print(f'Loading weights from {self.model_path}')
            # self.model.load_weights(self.model_path)
            self.model.summary()
            self.model_built = True
        pred = self.model.call(X, training=False).numpy()
        return pred, Y, Z[:, 0, -2], Z[:, :, -1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parity", type=int)
    parser.add_argument("-training_variables", "--training_variables")
    parser.add_argument("-params_json", "--params_json")
    parser.add_argument("-f", "--file", nargs='+')
    parser.add_argument("-w", "--weights")
    #parser.add_argument("-output", "--output")

    args = parser.parse_args()
    file_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(file_dir)

    cfg_dir = os.path.join(base_dir, 'config')
    applier = ApplyTraining(args.params_json, os.path.join(cfg_dir, 'mean_std_red.json'),
                            os.path.join(cfg_dir, 'min_max_red.json'), args.training_variables)
    pred, Y, evt, cpp_scores = applier.apply(args.file[0], args.weights, args.parity)
    #np.save(args.output, pred)

    # save_frozen_graph('my_model.pb', applier.model)
