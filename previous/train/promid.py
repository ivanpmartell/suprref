import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, ProgressBar, Checkpoint
from skorch.helper import predefined_split
from skorch.dataset import Dataset

sys.path.append(os.path.join(sys.path[0], '..'))
from PROMID.module import PROMIDModule
from PROMID.dataset import PROMIDDataset
from PROMID.callbacks import AppendToDataset

model_folder = "models/promid/"
images_folder= "models/promid/images"
if not os.path.exists(images_folder):
    os.makedirs(images_folder)
# Binary(sigmoid): Use NeuralNetBinaryClassifier (!IMPORT IT), num_classes=1, criterion=BCEWithLogitsLoss, binary=True
# Multi(softmax): Use NeuralNetClassifier (!IMPORT IT), num_classes=2, criterion=CrossEntropyLoss, binary=False

print("Preprocessing: Started")
ds = PROMIDDataset(file="data/human_nonTATA_5000.fa", tss_loc=5000, binary=False, save_df=False)
print("Preprocessing: Done")
weight = torch.FloatTensor((0.5,0.5))
net = NeuralNetClassifier(module=PROMIDModule,
                          module__num_classes=2,
                          module__seqs_length=ds.seqs_length,
                          criterion=torch.nn.CrossEntropyLoss,
                          criterion__weight=weight,
                          max_epochs=5000,
                          lr=0.0001,
                          callbacks=[AppendToDataset(),
                                     ProgressBar(),
                                     Checkpoint(monitor=None,
                                                dirname=model_folder,
                                                f_params='model.pt')],
                          batch_size=32,
                          optimizer=torch.optim.Adam,
                          optimizer__weight_decay=1.5e-3,
                          device='cuda' if torch.cuda.is_available() else 'cpu',
                          train_split=predefined_split(ds.val_dataset))

print("Training: Started")
net.fit(ds, None)
print("Training: Done")