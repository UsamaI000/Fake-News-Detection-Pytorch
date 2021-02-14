# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import Config
import re

config = Config()

# METHOD TO PLOT ACCURACY AND LOSS CURVES
def plotCurves(stats, path):
    fig = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec2[0, 0])
    ax2 = fig.add_subplot(spec2[0, 1])
    for i in ['train_loss', 'valid_loss']: 
      ax1.plot(stats[i], label=i)
    ax1.legend()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    for j in ['train_acc', 'valid_acc']: 
      ax2.plot(100 * stats[j], label=j)
    ax2.legend()
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')

    fig.savefig(path+'/Metrics.png')

# METHOD TO PLOT CONFUSION MATRIX USING HEATMAP SEABORN
def Confusion_matrix(original, prediction, title='Confusion matrix', labels=['0','1'], path=None):
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot()
    ax.set_title(title)
    cm = confusion_matrix(original, prediction)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt="d", xticklabels=labels, yticklabels=labels)
    if path != None: plt.savefig(path)
    else: plt.show()


# METHOD TO PRE-PROCESS DATA
def preprocess(text, stem=False, stop_w=False):
    # Remove link,user and special characters
    text = re.sub(config.TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    if stop_w:
      for token in text.split():
          if token not in config.stop_words:
              if stem:
                  tokens.append(config.stemmer.stem(token))
              else:
                  tokens.append(token)
      txt = " ".join(tokens)
      return txt
    else: return text


# METHOD TO OBTAIN DATA LOADER
def Dataloader(data_x, data_y, batch_size):
    data = TensorDataset(torch.from_numpy(data_x), torch.from_numpy(data_y))
    batch_size = batch_size
    data_loader = DataLoader(data, shuffle=True, batch_size=batch_size, drop_last=True)
    return data_loader