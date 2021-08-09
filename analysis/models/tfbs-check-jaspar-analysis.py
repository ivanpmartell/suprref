import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint

sys.path.append("..")
from tfbs_check.modules import JASPARModule as torch_module
from tfbs_check.dataset import TFBSDataset
from tfbs_check.helper import Helper

import seqlogo

helper_class = Helper()
helper_class.add_variable("model", "models/tfbs_check_jaspar_polii/results/model.pt")
helper_class.add_variable("binary", False)
helper_class.add_variable("matrix_type", 'pwm')
helper_class.add_variable("stride", 1)

matrices = helper_class.get_jaspar_matrices({None: 'POLII'})
max_filter_len, _ = Helper.get_filters(matrices, helper_class.get_variable("matrix_type"))
helper_class.add_variable("padding", max_filter_len-1)

if helper_class.get_variable("padding") >= max_filter_len:
    raise Exception("Invalid padding")
    
if helper_class.get_variable("stride") > max_filter_len:
    raise Exception("Invalid stride")

helper_class.add_variable("promoters", Helper.PROMOTER_FILE)
helper_class.add_variable("non-promoters", Helper.NONPROMOTER_FILE)
promoter_path = os.path.abspath(helper_class.get_variable("promoters"))
nonpromoter_path = os.path.abspath(helper_class.get_variable("non-promoters"))

ds = TFBSDataset(file=promoter_path, neg_file=nonpromoter_path, binary=False, save_df=False)
helper_class.add_variable("length", ds.seqs_length)

def analyse_weights(title, helper_class, save_location, filter_weights, filter_names, aggregate_function, weights1=None, weights2=None):
    print(title)
    Path(save_location).mkdir(parents=True, exist_ok=True)

    conv_result_length = len(weights1) // len(filter_weights)
    for position in range(conv_result_length):
        combined_filter_weights = np.zeros_like(filter_weights[0])
        for idx, w in enumerate(filter_weights):
            balanced_weights = aggregate_function(idx, w, position, conv_result_length, weights1, weights2)
            combined_filter_weights += balanced_weights
            weights = Helper.softmax(balanced_weights)
            ppm = weights.T
            ppm_logo = seqlogo.Ppm(Helper.sum_one(ppm))
            file_location = os.path.join(save_location,"position{0}_filter-{1}.svg".format(position, filter_names[idx]))
            seqlogo.seqlogo(ppm_logo, ic_scale = True, format = 'svg', size = 'large', filename = file_location)
        weights = Helper.softmax(combined_filter_weights)
        ppm = weights.T
        ppm_logo = seqlogo.Ppm(Helper.sum_one(ppm))
        file_location = os.path.join(save_location,"position{0}_combined.svg".format(position))
        seqlogo.seqlogo(ppm_logo, ic_scale = True, format = 'svg', size = 'large', filename = file_location)


fullpath = os.path.abspath(helper_class.get_variable("model"))
model_folder = os.path.dirname(fullpath)
cp = Checkpoint(dirname=model_folder, f_params=os.path.basename(fullpath))
# Binary(sigmoid): Use NeuralNetBinaryClassifier, num_classes=1, criterion=BCEWithLogitsLoss, binary=True
# Multi(softmax): Use NeuralNetClassifier, num_classes=2, criterion=CrossEntropyLoss, binary=False
    
net = NeuralNetClassifier(module=torch_module,
                          module__num_classes=1 if helper_class.get_variable("binary") else 2,
                          module__seqs_length=helper_class.get_variable("length"),
                          module__matrix_type=helper_class.get_variable("matrix_type"),
                          module__matrices=matrices,
                          module__stride=helper_class.get_variable("stride"),
                          module__padding=helper_class.get_variable("padding"))
net.initialize()
print("Network Initialized")
net.load_params(checkpoint=cp)
print("Model Loaded")

print("Detaching weights from network")
output_weights = net.module_.out.weight.detach().numpy().astype(float)
promoter_weights = output_weights[helper_class.get_LABEL_dict()['Promoter']]
non_promoter_weights = output_weights[helper_class.get_LABEL_dict()['Non-Promoter']]
if (len(promoter_weights) != len(non_promoter_weights)):
    raise Exception("Something is wrong with the weights")

filter_weights = np.squeeze(net.module_.filters.detach().numpy().astype(float))
if filter_weights.ndim < 3:
    filter_weights = np.expand_dims(filter_weights, axis=0)
filter_names = net.module_.filter_names

title = "Analyzing non-promoter weights"
save_location = os.path.join(model_folder,"seqlogo","nonpromoter-only")
analyse_weights(title, helper_class, save_location, filter_weights, filter_names, Helper.aggregate_weights, weights1=non_promoter_weights)

title = "Analyzing promoter weights - non-promoter weights"
save_location = os.path.join(model_folder,"seqlogo","promoter-nonpromoter")
analyse_weights(title, helper_class, save_location, filter_weights, filter_names, Helper.aggregate_weights, promoter_weights, non_promoter_weights)

title = "Analyzing non-promoter weights - promoter weights"
save_location = os.path.join(model_folder,"seqlogo","nonpromoter-promoter")
analyse_weights(title, helper_class, save_location, filter_weights, filter_names, Helper.aggregate_weights, non_promoter_weights, promoter_weights)

title = "Analyzing promoter weights"
save_location = os.path.join(model_folder,"seqlogo","promoter-only")
analyse_weights(title, helper_class, save_location, filter_weights, filter_names, Helper.aggregate_weights, weights1=promoter_weights)