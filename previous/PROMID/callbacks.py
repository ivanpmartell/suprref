from skorch.callbacks import Callback
import torch

class AppendToDataset(Callback):
    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None):
        if net.history[-1]['epoch'] % 10 == 0:
            net.module_.eval()
            with torch.no_grad():
                dataset_train.append_false_positive_seqs(net)
            prom, nonprom = dataset_train.get_class_amounts()
            total = prom + nonprom
            weight = torch.FloatTensor((nonprom/total, prom/total))
            net.set_params(criterion__weight=weight)
            print("New promoter weights: %f, New non-promoter weights: %f" % (nonprom/total, prom/total))