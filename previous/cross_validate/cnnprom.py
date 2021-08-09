import os
import sys
import torch
import argparse
import numpy as np
from numpy import average
from tqdm import tqdm
from skorch import NeuralNetClassifier, NeuralNetBinaryClassifier
from skorch.dataset import CVSplit
from skorch.callbacks import EarlyStopping, ProgressBar, Checkpoint
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, confusion_matrix, make_scorer
from sklearn.model_selection import cross_validate, cross_val_score

sys.path.append(os.path.join(sys.path[0], '..'))
from CNNPROM.module import CNNPROMModule
from CNNPROM.dataset import CNNPROMDataset

###########################################
# Command line interface
this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
default_out = this_dir
default_input = "data/human_complete.fa"
default_neg = ""
default_neg_size = 27731 + 8256
default_num_channels = 300
default_pool_size = 231
num_folds=10

parser = argparse.ArgumentParser(description=r"This script will test a model's performance with CNNProm dataset")
parser.add_argument('-binary', 
        type=bool, 
        help='For model: a 1 neuron sigmoid output if set, otherwise a 2 neuron softmax output',
        default=False)
parser.add_argument('--output',
        type = str,
        help = f'Path for desired output folder. Default: {default_out}. '
        'The output file is a csv with the sequences tested, their true labels, and the predictions by the model',
        default = default_out
        )
parser.add_argument('--input',
        type = str,
        help = f'Path to desired input file. Default: {default_input}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_input
        )
parser.add_argument('--neg_file',
        type = str,
        help = f'Path to desired annotations file. Default: {default_neg}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_neg
        )
parser.add_argument('--neg_sample',
        type = int,
        help = f'Path to desired annotations file. Default: {default_neg_size}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_neg_size
        )
parser.add_argument('--num_channels',
        type = int,
        help = f'Path to desired annotations file. Default: {default_num_channels}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_num_channels
        )
parser.add_argument('--pool_size',
        type = int,
        help = f'Path to desired annotations file. Default: {default_pool_size}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_pool_size
        )
args = parser.parse_args()
###########################################

model_folder = args.output
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
# Binary(sigmoid): Use NeuralNetBinaryClassifier (!IMPORT IT), num_classes=1, criterion=BCEWithLogitsLoss, binary=True
# Multi(softmax): Use NeuralNetClassifier (!IMPORT IT), num_classes=2, criterion=CrossEntropyLoss, binary=False

neg_f = None
if(args.neg_file != ''):
    neg_f = args.neg_file
if(args.binary):
    nc = 1
    crit = torch.nn.BCEWithLogitsLoss
    cls = NeuralNetBinaryClassifier
else:
    nc = 2
    crit = torch.nn.CrossEntropyLoss
    cls = NeuralNetClassifier
ds = CNNPROMDataset(file=args.input, neg_file=neg_f, num_negatives=args.neg_sample, binary=args.binary, save_df=None, drop_dups=False)
print("Preprocessing: Preparing for stratified sampling")
data_list = [(x, y) for x, y in tqdm(iter(ds))]
X = np.array([col[0] for col in data_list], dtype=np.float32)
y = np.array([col[1] for col in data_list], dtype=np.int64)
print("Preprocessing: Done")
net = cls(module=CNNPROMModule,
                          module__num_classes=nc,
                          module__seqs_length=ds.seqs_length,
                          module__num_channels=args.num_channels,
                          module__pool_kernel=args.pool_size,
                          criterion=crit,
                          max_epochs=50,
                          lr=0.005,
                          callbacks=[EarlyStopping(patience=10),
                                     ProgressBar(),
                                     Checkpoint(dirname=model_folder,
                                                f_params='model.pt')],
                          batch_size=16,
                          optimizer=torch.optim.Adam,
                          train_split=CVSplit(cv=0.1,stratified=True),
                          device='cuda' if torch.cuda.is_available() else 'cpu')

print("Cross Validation: Started")
#scoring metrics can be modified. Predefined metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
out_path = os.path.join(model_folder, "cv_results.txt")
with open(out_path, 'w') as f:
    f.write("tn,fp,fn,tp,sn,sp,ppv,mcc,acc\n")
def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    tn = cm[0, 0]
    tp = cm[1, 1]
    fn = cm[1, 0]
    fp = cm[0, 1]
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    precision = tp/(tp+fp)
    mcc = matthews_corrcoef(y, y_pred)
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    with open(out_path, 'a') as f:
        f.write(f'{tn},{fp},{fn},{tp},{sensitivity},{specificity},{precision},{mcc},{accuracy}\n')
    return {'tn': tn , 'fp': fp,
            'fn': fn, 'tp': tp,
            'sensitivity': sensitivity, 'specificity': specificity,
            'precision': precision, 'mcc': mcc, 'accuracy': accuracy }
results = cross_validate(net, X, y, scoring=confusion_matrix_scorer, cv=num_folds, verbose=1)
print("Cross Validation: Done")