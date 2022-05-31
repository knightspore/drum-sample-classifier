# Imports
import torch, wandb, time
import torch.nn as nn
from torch.utils.data import DataLoader,random_split,SubsetRandomSampler, ConcatDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from model import AudioClassifier
from util import WavDataSet, SoundDS, PlotSpectrogram, predict, classes, classes_reverse
from training_standard import training, inference  
from training_k_fold import train_epoch, valid_epoch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
df = pd.read_csv("./data/edm_no_loops.csv")
df = df[['path', 'class']]
myds = SoundDS(df)
num_items = len(myds)
num_train = round(num_items * 0.67)
num_val = num_items-num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])
batch_size = 64
num_epochs = 1000
lr = 2e-4
max_lr = 1e-2

model = AudioClassifier()
model.to(device)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)

training(model, train_dl, device=device, num_epochs=num_epochs, lr=lr, max_lr=max_lr, logger=None)
inference(model, val_dl, device=device, logger=None)

torch.save(model, f'models/model_lr{lr}_mlr{max_lr}_e{num_epochs}_b{batch_size}.pt')
