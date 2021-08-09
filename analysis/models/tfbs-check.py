import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from tfbs_check.modules import CNNModule as torch_module
from tfbs_check.dataset import TFBSDataset
from tfbs_check.helper import Helper
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
from skorch.callbacks import EarlyStopping, ProgressBar, Checkpoint, EpochScoring

        

model_folder = "models/tfbs_check_nonjaspar/results/"
abs_folder = os.path.abspath(model_folder)
Path(abs_folder).mkdir(parents=True, exist_ok=True)
# Binary(sigmoid): Use NeuralNetBinaryClassifier, num_classes=1, criterion=BCEWithLogitsLoss, binary=True
# Multi(softmax): Use NeuralNetClassifier, num_classes=2, criterion=CrossEntropyLoss, binary=False

ds = TFBSDataset(file=Helper.PROMOTER_FILE, neg_file=Helper.NONPROMOTER_FILE, binary=False, save_df=False)
print("Preprocessing: Preparing for stratified sampling")
y_train = np.array([y for _, y in tqdm(iter(ds))])
print("Preprocessing: Done")
recall = EpochScoring(scoring='recall', lower_is_better=False)
precision = EpochScoring(scoring='precision', lower_is_better=False)
net = NeuralNetClassifier(module=torch_module,
                          module__num_classes=2,
                          module__seqs_length=ds.seqs_length,
                          criterion=torch.nn.CrossEntropyLoss,
                          #criterion__weight=torch.Tensor(np.array([0.2,0.8])),
                          max_epochs=100,
                          lr=0.01,
                          callbacks=[EarlyStopping(patience=20),
                                     ProgressBar(),
                                     Checkpoint(dirname=model_folder,
                                                f_params='model.pt'),
                                     recall, precision],
                          batch_size=16,
                          optimizer=torch.optim.Adam,
                          train_split=CVSplit(cv=0.2,stratified=True))
print("Training: Started")
net.fit(ds, y_train)
print("Training: Done")
