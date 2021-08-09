import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from skorch.dataset import CVSplit
from skorch import NeuralNetClassifier, NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping, ProgressBar, Checkpoint

sys.path.append(os.path.join(sys.path[0], '..'))
from ICNN.module import ICNNModule
from ICNN.dataset import ICNNDataset

###########################################
# Command line interface
this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
default_out = os.path.join(os.path.dirname(this_dir), "results.csv")
default_input = "data/human_representative.fa"
default_pos_size = 7156
default_neg = "data/bdgp"
default_mod = "models/icnn/model.pt"

parser = argparse.ArgumentParser(description=r"This script will test a model's performance with ICNN dataset")
parser.add_argument('-binary', 
        type=bool, 
        help='For model: a 1 neuron sigmoid output if set, otherwise a 2 neuron softmax output',
        default=False)
parser.add_argument('--output',
        type = str,
        help = f'Path for desired output file. Default: {default_out}. '
        'The output file is a csv with the sequences tested, their true labels, and the predictions by the model',
        default = default_out
        )
parser.add_argument('--input',
        type = str,
        help = f'Path to desired input file. Default: {default_input}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_input
        )
parser.add_argument('--neg_folder',
        type = str,
        help = f'Path to desired annotations file. Default: {default_neg}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_neg
        )
parser.add_argument('--pos_sample',
        type = int,
        help = f'Path to desired annotations file. Default: {default_pos_size}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_pos_size
        )
args = parser.parse_args()
###########################################

model_folder = args.output
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
# Binary(sigmoid): Use NeuralNetBinaryClassifier (!IMPORT IT), num_classes=1, binary=True
# Multi(softmax): Use NeuralNetClassifier (!IMPORT IT), num_classes=2, binary=False

if(args.binary):
    nc = 1
    crit = torch.nn.BCEWithLogitsLoss
    cls = NeuralNetBinaryClassifier
else:
    nc = 2
    crit = torch.nn.CrossEntropyLoss
    cls = NeuralNetClassifier
ds = ICNNDataset(file=args.input, neg_folder=args.neg_folder, num_positives=args.pos_sample, binary=args.binary, save_df=None, drop_dups=False)
print("Preprocessing: Preparing for stratified sampling")
y_train = np.array([y for _, y in tqdm(iter(ds))])
print("Preprocessing: Done")
net = cls(module=ICNNModule,
                          module__num_classes=nc,
                          module__elements_length=ds.elements_length,
                          module__non_elements_length=ds.non_elements_length,
                          criterion=crit,
                          max_epochs=50,
                          lr=0.001,
                          callbacks=[EarlyStopping(patience=10),
                                     ProgressBar(),
                                     Checkpoint(dirname=model_folder,
                                                f_params='model.pt')],
                          train_split=CVSplit(cv=0.1,stratified=True),
                          batch_size=8,
                          optimizer=torch.optim.SGD,
                          optimizer__momentum=0.90,
                          device='cuda' if torch.cuda.is_available() else 'cpu')
print("Training: Started")
net.fit(ds, y_train)
print("Training: Done")