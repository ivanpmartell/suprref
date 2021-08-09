import os
from CNNPROM.dataset import CNNPROMDataset
from ICNN.dataset import ICNNDataset
from DPROM.dataset import DPROMDataset

def create_folder_ifne(model_folder):
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

#NO DUPLICATES
create_folder_ifne('datasets/dprom_TATA')
DPROMDataset(file="data/human_TATA.fa", neg_file=None, binary=False, save_df="dprom_TATA", test_set=True, split_ratio=0.2)
create_folder_ifne('datasets/dprom_nonTATA')
DPROMDataset(file="data/human_nonTATA.fa", neg_file=None, binary=False, save_df="dprom_nonTATA", test_set=True, split_ratio=0.2)
create_folder_ifne('datasets/dprom_complete')
DPROMDataset(file="data/human_complete.fa", neg_file=None, binary=False, save_df="dprom_complete", test_set=True, split_ratio=0.2)

create_folder_ifne('datasets/cnnprom_TATA')
CNNPROMDataset(file="data/human_TATA.fa", neg_file=None , num_negatives=8256, binary=False, save_df="cnnprom_TATA", num_positives=1426, test_set=True, split_ratio=0.2)
create_folder_ifne('datasets/cnnprom_nonTATA')
CNNPROMDataset(file="data/human_nonTATA.fa", neg_file=None , num_negatives=27731, num_positives=19811, binary=False, save_df="cnnprom_nonTATA", test_set=True, split_ratio=0.2)
create_folder_ifne('datasets/cnnprom_complete')
CNNPROMDataset(file="data/human_complete.fa", neg_file=None , num_negatives=8256 + 27731, num_positives=1426 + 19811, binary=False, save_df="cnnprom_complete", test_set=True, split_ratio=0.2)

create_folder_ifne('datasets/icnn_complete')
ICNNDataset(file="data/human_complete.fa", neg_folder="data/bdgp", num_positives=7156, binary=False, save_df="icnn_complete", test_set=True, split_ratio=0.2)

#WITH DUPLICATES
create_folder_ifne('datasets_dups/dprom_TATA')
DPROMDataset(file="data/human_TATA.fa", neg_file=None, binary=False, save_df="dprom_TATA", test_set=True, split_ratio=0.2, drop_dups=False, dataset_folder="datasets_dups")
create_folder_ifne('datasets_dups/dprom_nonTATA')
DPROMDataset(file="data/human_nonTATA.fa", neg_file=None, binary=False, save_df="dprom_nonTATA", test_set=True, split_ratio=0.2, drop_dups=False, dataset_folder="datasets_dups")
create_folder_ifne('datasets_dups/dprom_complete')
DPROMDataset(file="data/human_complete.fa", neg_file=None, binary=False, save_df="dprom_complete", test_set=True, split_ratio=0.2, drop_dups=False, dataset_folder="datasets_dups")

create_folder_ifne('datasets_dups/cnnprom_TATA')
CNNPROMDataset(file="data/human_TATA.fa", neg_file=None , num_negatives=8256, binary=False, save_df="cnnprom_TATA", num_positives=1426, test_set=True, split_ratio=0.2, drop_dups=False, dataset_folder="datasets_dups")
create_folder_ifne('datasets_dups/cnnprom_nonTATA')
CNNPROMDataset(file="data/human_nonTATA.fa", neg_file=None , num_negatives=27731, num_positives=19811, binary=False, save_df="cnnprom_nonTATA", test_set=True, split_ratio=0.2, drop_dups=False, dataset_folder="datasets_dups")
create_folder_ifne('datasets_dups/cnnprom_complete')
CNNPROMDataset(file="data/human_complete.fa", neg_file=None , num_negatives=8256 + 27731, num_positives=1426 + 19811, binary=False, save_df="cnnprom_complete", test_set=True, split_ratio=0.2, drop_dups=False, dataset_folder="datasets_dups")

create_folder_ifne('datasets_dups/icnn_complete')
ICNNDataset(file="data/human_complete.fa", neg_folder="data/bdgp", num_positives=7156, binary=False, save_df="icnn_complete", test_set=True, split_ratio=0.2, drop_dups=False, dataset_folder="datasets_dups")