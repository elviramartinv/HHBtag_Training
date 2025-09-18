import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
import argparse

from Apply import ApplyTraining
import InputsProducer

# Define the custom metric MeanMetricWrapper
class MeanMetricWrapper(tf.keras.metrics.Metric):
    def __init__(self, name='mean_metric', **kwargs):
        super(MeanMetricWrapper, self).__init__(name=name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = custom_metric(y_true, y_pred)  
        self.total.assign_add(tf.reduce_sum(values))
        self.count.assign_add(tf.cast(tf.shape(values)[0], tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        base_config = super(MeanMetricWrapper, self).get_config()
        return {**base_config}

    @classmethod
    def from_config(cls, config):
        # Remove the 'fn' keyword argument from the config
        config.pop('fn', None)
        return cls(**config)

# Load the variables from the training_variables.json file
with open('../config/training_variables.json', 'r') as f:
    training_variables = json.load(f)

file_name = "/afs/cern.ch/user/e/emartinv/public/Fork_Framework/Framework/training_skim/GluGluToHHTo2B2Tau_node_2.root"
parity = 0  
training_variables_str = "../config/training_variables.json"

data = InputsProducer.CreateRootDF(file_name, parity, False, True)
X, Y, Z, var_pos, var_pos_z, var_name = InputsProducer.CreateXY(data, training_variables_str)

# Event variables (evt_vars) and jet variables (jet_vars)
evt_vars = training_variables['evt_vars']
jet_vars = training_variables['jet_vars']

all_features = evt_vars + [var.format(i) for i in range(6) for var in jet_vars]

selected_features = all_features[:10]

# Filter the data to include only the selected features
selected_feature_indices = [all_features.index(feature) for feature in selected_features]
# X_selected = X[:, :, selected_feature_indices]

# Load the model from the hhmodel_PNet/model file with custom_objects
custom_objects = {'MeanMetricWrapper': MeanMetricWrapper}
model = load_model('hhmodel_PNet/model', custom_objects=custom_objects)
X = X.reshape((X.shape[0], -1))
# Calculate permutation_importance
result = permutation_importance(model, X, Y, n_repeats=10, random_state=123)

# Get feature importances
feature_importance = result.importances_mean

# Visualize feature importances in a bar chart
plt.figure(figsize=(10, 6))
plt.bar(range(len(selected_features)), feature_importance, tick_label=selected_features)
plt.xticks(rotation=90)
plt.title('Permutation Feature Importance')
plt.tight_layout()
plt.show()
