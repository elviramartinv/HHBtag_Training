import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import InputsProducer as InputsProducer
import ParametrizedModel as pm

data = pd.read_csv('ggf_model/history.csv')

# 'loss', 'val_loss', 'sel_acc_2' y 'val_sel_acc_2'
loss = data['loss']
val_loss = data['val_loss']
accuracy = data['sel_acc_2']
val_accuracy = data['val_sel_acc_2']
epochs = data['epoch']

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, label='Trai', color='blue')
plt.plot(epochs, val_loss, label='Val', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, label='Train', color='blue')
plt.plot(epochs, val_accuracy, label='Val', color='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Purity')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('../output/model/training_evaluation.pdf', dpi=300)


