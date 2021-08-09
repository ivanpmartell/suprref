import os
import torch
import numpy as np
from tqdm import tqdm
from ICNN.dataset import ICNNDataset
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

model_folder = "models/icnn/svm/"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

ds = ICNNDataset(file="data/human_representative.fa", neg_folder="data/bdgp", num_positives=7156, binary=False, save_df=True)
print("Preprocessing: Preparing for stratified sampling")
data = [[X, y] for X, y in tqdm(iter(ds))]
X, y_train = zip(*data)
print("Preprocessing: Done")
model = LinearSVC(max_iter=5000, verbose=1)
print("Training: Started")
print(cross_val_score(model, X, y_train, cv=10))
print("Training: Done")