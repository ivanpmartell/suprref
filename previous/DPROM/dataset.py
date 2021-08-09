import pathlib
import numpy as np
from Bio import SeqIO
import pandas as pd
import random
from skorch.dataset import Dataset
from tqdm import tqdm

dna_list = ["A", "G", "C", "T"]

class DPROMDataset(Dataset):
    seqs_length: int
    y_type: np.dtype = np.int64
    dataframe: pd.DataFrame
    dna_dict: dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    lbl_dict: dict = {'Non-Promoter': 0, 'Promoter': 1}

    def __init__(self, file, neg_file, binary, save_df=None, test_set=False, split_ratio=0.30, drop_dups=True, dataset_folder="datasets"):
        if('csv' in pathlib.Path(file).suffix):
            self.load_dataframe(file)
        else:
            seqs = self.load_file(file)
            self.seqs_length = len(seqs[0])
            df = pd.DataFrame(seqs, columns=['sequence'])
            if(drop_dups):
                df.drop_duplicates(inplace=True)
            df['label'] = self.lbl_dict['Promoter']
            if(neg_file is not None):
                neg_seqs = self.load_file(neg_file)
                neg_seqs_length = len(neg_seqs[0])
                if(self.seqs_length != neg_seqs_length):
                    raise Exception(r"Promoter and Non-Promoter sequence lengths don't match")
            else:
                print("Preprocessing: Creating the negative sequences")
                neg_seqs = []
                for seq in tqdm(df['sequence']):
                    neg_seq = self.create_negative_seq(seq)
                    neg_seqs.append(neg_seq)
            neg_df = pd.DataFrame(neg_seqs, columns=['sequence'])
            if(drop_dups):
                neg_df.drop_duplicates(inplace=True)
            neg_df['label'] = self.lbl_dict['Non-Promoter']
            self.dataframe = df.append(neg_df, ignore_index=True)
        if(binary):
            self.y_type = np.float32
        if(save_df is not None):
            self.save_dataframe(f'{dataset_folder}/{save_df}/dataframe.csv')
        if test_set:
            from sklearn.model_selection import train_test_split
            train, test = train_test_split(self.dataframe, stratify=self.dataframe["label"], test_size=split_ratio)
            train.to_csv(f'{dataset_folder}/{save_df}/train.csv',index=False)
            test.to_csv(f'{dataset_folder}/{save_df}/test.csv',index=False)

    def load_file(self, file):
        records = []
        with open(file, 'rU') as fasta_file:
            for record in SeqIO.parse(fasta_file, 'fasta'):
                r_seq = record.seq._data
                if 'N' not in r_seq:
                    records.append(r_seq)
        return records

    def load_dataframe(self, file):
        self.dataframe = pd.read_csv(file)
        example = self.dataframe.iloc[0]
        self.seqs_length = len(example.sequence)
    
    def save_dataframe(self, file):
        print('Saving dataframe to: %s' % file)
        self.dataframe.to_csv(file, index=False)
    
    def create_negative_seq(self, seq):
        split = len(seq) // 20
        split_seq = [seq[i:i+split] for i in range(0, len(seq), split)]
        indices = np.arange(20)
        np.random.shuffle(indices)
        for i in range(12):
            split_seq[indices[i]] = self.get_random_dna_seq(split_seq[indices[i]])
        return ''.join(split_seq)

    @staticmethod
    def get_random_dna_seq(org_seq):
        new_seq = ''
        for _ in range(len(org_seq)):
            new_seq += random.choice(dna_list)
        return new_seq

    def one_hot_encoder(self, seq):
        one_hot = np.zeros((len(self.dna_dict), len(seq)), dtype=np.float32)
        for idx, token in enumerate(seq):
            one_hot[self.dna_dict[token], idx] = 1
        return one_hot
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        x = self.one_hot_encoder(row.sequence.upper())
        return x, np.array(row.label, dtype=self.y_type)
    
    def __len__(self):
        return len(self.dataframe)