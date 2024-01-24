# python training_eval.py model_name
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import InputsProducer as InputsProducer
import ParametrizedModel as pm


parser = argparse.ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()

model_path = os.path.join(args.model, 'history.csv')
data = pd.read_csv(model_path)

# 'loss', 'val_loss', 'sel_acc_2' y 'val_sel_acc_2'
loss = data['loss']
val_loss = data['val_loss']
accuracy = data['sel_acc_2']
val_accuracy = data['val_sel_acc_2']
epochs = data['epoch']

plt.figure(figsize=(6, 4))
# plt.subplot(1, 2, 1)
plt.plot(epochs, loss, label='Train', color='blue')
plt.plot(epochs, val_loss, label='Val', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
output_acc = os.path.join(args.model, 'training_loss_evaluation.pdf')
plt.tight_layout()
plt.savefig(output_acc, dpi=300)
plt.close()

plt.figure(figsize=(6,4))
# plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, label='Train', color='blue')
plt.plot(epochs, val_accuracy, label='Val', color='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Purity')
plt.legend()
plt.grid(True)
output_loss = os.path.join(args.model, 'training_acc_evaluation.pdf')
plt.tight_layout()
plt.savefig(output_loss, dpi=300)
plt.close()




