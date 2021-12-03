import torch
import numpy as np
from tqdm import tqdm
from skorch import NeuralNetClassifier, NeuralNetBinaryClassifier, NeuralNet
from skorch.callbacks import EarlyStopping, ProgressBar, Checkpoint

try:
    from ..helper import Helper
    from .dataset import IODataset
except:
    from helper import Helper
    from IO.dataset import IODataset

class IOTrainer:
    _helper: Helper
    _dataset: IODataset
    _net: NeuralNet

    def __init__(self, helper, output_func, module, module_args, max_epochs, learning_rate, batch_size, optimizer, optimizer_args, patience):
        self._helper = helper
        window_size = self._helper.CONF_DICT[Helper._DICTKEY_CONFIGURATION][Helper._DICTKEY_SEQ_UPSTREAM_LEN] + self._helper.CONF_DICT[Helper._DICTKEY_CONFIGURATION][Helper._DICTKEY_SEQ_DOWNSTREAM_LEN]
        if output_func == 'sigmoid':
            y_type = np.float32
            num_out_neurons = 1
            criterion= torch.nn.BCEWithLogitsLoss
            classifier = NeuralNetBinaryClassifier
        elif output_func == 'softmax':
            y_type = np.int64
            num_out_neurons = 2
            criterion= torch.nn.CrossEntropyLoss
            classifier = NeuralNetClassifier
        self._dataset = IODataset(self._helper, create_dataset=False,  y_type=y_type)
        self._net = classifier(module=module, module__num_classes=num_out_neurons,
                         module__seqs_length=window_size,
                         criterion=criterion, max_epochs=50, lr=learning_rate, batch_size=batch_size,
                         device='cuda' if torch.cuda.is_available() else 'cpu',
                         optimizer=getattr(torch.optim, optimizer),
                         callbacks=[EarlyStopping(patience=5),
                                    ProgressBar(),
                                    Checkpoint(dirname=self._helper.CONF_DICT[Helper._DICTKEY_EXPERIMENT_FOLDER],
                                               f_params='model.pt')])
        print("Preprocessing: Preparing for stratified sampling")
        self._y_train = self._dataset.get_y()
        print("Preprocessing: Done")

    def fit(self):
        print("Training: Started")
        self._net.fit(self._dataset, self._y_train)
        print("Training: Done")
