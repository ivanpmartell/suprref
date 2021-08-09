import pathlib
import numpy as np
from Bio import SeqIO
import pandas as pd
import random
from skorch.dataset import Dataset
from .helper import Helper
from tqdm import tqdm

class TFBSDataset(Dataset):
    seqs_start: int = 200
    seqs_length: int
    y_type: np.dtype = np.int64
    dataframe: pd.DataFrame
    helper_class: Helper

    def __init__(self, file, neg_file, binary, save_df):
        self.helper_class = Helper()
        seqs = self.load_file(file)
        self.seqs_length = len(seqs[0])
        df = pd.DataFrame(seqs, columns=['sequence'])
        df['label'] = self.helper_class.get_LABEL_dict()['Promoter']

        if(neg_file is not None):
            neg_seqs = self.load_file(neg_file)
            neg_seqs_length = len(neg_seqs[0])
            if(self.seqs_length != neg_seqs_length):
                raise Exception(r"Promoter and Non-Promoter sequence lengths don't match")
            neg_df = pd.DataFrame(neg_seqs, columns=['sequence'])
            neg_df['label'] = self.helper_class.get_LABEL_dict()['Non-Promoter']
            self.dataframe = df.append(neg_df, ignore_index=True)
        if(binary):
            self.y_type = np.float32
        if(save_df):
            self.save_dataframe('models/tfbs_check/dataframe.csv')

    def load_file(self, file):
        records = []
        with open(file, 'rU') as fasta_file:
            for record in SeqIO.parse(fasta_file, 'fasta'):
                r_seq = self.sequence_extractor(record.seq._data.upper())
                if 'N' not in r_seq:
                    records.append(r_seq)
        return records

    def save_dataframe(self, file):
        print('Saving dataframe to: %s' % file)
        self.dataframe.to_csv(file, index=False)

    def one_hot_encoder(self, seq):
        one_hot = np.zeros((len(self.helper_class.get_DNA_dict()), len(seq)), dtype=np.float32)
        for idx, token in enumerate(seq):
            one_hot[self.helper_class.get_DNA_dict()[token], idx] = 1
        return one_hot

    def TFIIB_TATA_Box_extractor(self, seq):
        return seq[self.seqs_start -45:self.seqs_start -20 + 1]

    def sequence_extractor(self, seq):
        #element = self.TFIIB_TATA_Box_extractor(seq)
        return seq
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        x = self.one_hot_encoder(row.sequence)
        return x, np.array(row.label, dtype=self.y_type)
    
    def __len__(self):
        return len(self.dataframe)