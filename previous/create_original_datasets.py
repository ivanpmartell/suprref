import os
from CNNPROM.dataset import CNNPROMDataset
from ICNN.dataset import ICNNDataset
from DPROM.dataset import DPROMDataset

def create_folder_ifne(model_folder):
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

create_folder_ifne('datasets_original/cnnprom_nonTATA')
CNNPROMDataset(file="data/CNNPromoterData/human_non_tata.fa", neg_file="data/CNNPromoterData/human_nonprom_big.fa" , num_negatives=None, binary=False, save_df="cnnprom_nonTATA", num_positives=None, test_set=True, split_ratio=0.2, original=True, drop_dups=False, dataset_folder="datasets_original")

create_folder_ifne('datasets_original_clean/cnnprom_nonTATA')
CNNPROMDataset(file="data/CNNPromoterData/human_non_tata.fa", neg_file="data/CNNPromoterData/human_nonprom_big.fa" , num_negatives=None, binary=False, save_df="cnnprom_nonTATA", num_positives=None, test_set=True, split_ratio=0.2, original=True, drop_dups=True, dataset_folder="datasets_original_clean")
