import os
import numpy as np
from Bio import SeqIO
import pandas as pd
import random
from skorch.dataset import Dataset
from tqdm import tqdm

class OURDataset(Dataset):
    seqs_length: int
    dataframe: pd.DataFrame
    y_type: np.dtype = np.int64
    dna_dict: dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3,
                      'a': 0, 't': 1, 'g': 2, 'c': 3}
    dna_dict_count: int
    lbl_dict: dict = {'Non-Promoter': 0, 'Promoter': 1}
    chr_dict: dict = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
                      '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12,
                      '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18,
                      '19': 19, '20': 20, '21': 21, '22': 22, 'X': 23, 'Y': 24}

    def __init__(self, annotations_file, chromosome_name, promoter_upstream_length, promoter_downstream_length, input_length, threshold, stride, binary):
        self.seqs_length = input_length
        df = self.create_data(annotations_file, chromosome_name, promoter_upstream_length, promoter_downstream_length, input_length, threshold, stride)
        self.dataframe = pd.DataFrame(df, columns=['sequence', 'label'])
        if(binary):
            self.y_type = np.float32
        
    
    def create_data(self, annotations_file, chromosome_name, promoter_upstream_length, promoter_downstream_length, input_length, threshold, stride):
        self.dna_dict_count = len(set(self.dna_dict.values()))
        if(threshold is None):
            self.threshold = promoter_upstream_length + promoter_downstream_length + 1
        else:
            self.threshold = threshold
        annotations = pd.read_csv(annotations_file, sep='\t', names=["Id", "Type", "Position", "Strand", "Chromosome", "Gene"])
        annotations['Chromosome'] = annotations.Id.str[7:9].astype(int)
        
        chromosome_file = 'data/human_chrs/chr%s.fa' % chromosome_name
        with open(chromosome_file, 'rU') as file:
            chromosome = SeqIO.read(file, 'fasta').seq._data
        chromosome = str.encode(chromosome)
        chromosome = np.frombuffer(chromosome, dtype='S1')
        chr_length = len(chromosome)
        windows = (chr_length // stride) - (input_length // stride) + 1
        labels = np.zeros(chr_length, dtype=np.int64)
        chr_annotations = annotations[annotations['Chromosome'] == self.chr_dict[chromosome_name]]
        for _, row in tqdm(chr_annotations.iterrows(), total=len(chr_annotations)):
            if row['Strand'] == '-':
                start = row['Position'] - promoter_downstream_length - 1
                end = row['Position'] + promoter_upstream_length
                promoter = chromosome[start:end].tostring().decode('UTF-8')
                promoter = promoter.upper()
                promoter = self.create_antisense_strand(promoter)
            else:
                start = row['Position'] - promoter_upstream_length - 1
                end = row['Position'] + promoter_downstream_length
                promoter = chromosome[start:end].tostring().decode('UTF-8')
                promoter = promoter.upper()
            labels[start: end] = 1
        
        dataframe = []
        for idx in tqdm(range(windows)):
            start = idx * stride
            end = start + input_length
            if(end >= chr_length):
                start = chr_length - input_length
                end = start + input_length
            sequence = chromosome[start:end].tostring().decode('UTF-8')
            try:
                self.one_hot_encoder(sequence)
            except:
                continue
            sequence_label = labels[start:end]
            if(np.sum(sequence_label) >= self.threshold):
                y = self.lbl_dict['Promoter']
            else:
                y = self.lbl_dict['Non-Promoter']
            dataframe.append([sequence, y])
        return dataframe

    reversal = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    def create_antisense_strand(self, org_seq):
        negs = []
        for letter in org_seq[::-1]:
            negs.append(self.reversal[letter])
        result = ''.join(negs)
        return result

    def one_hot_encoder(self, seq):
        one_hot = np.zeros((self.dna_dict_count, len(seq)), dtype=np.float32)
        for idx, token in enumerate(seq):
            one_hot[self.dna_dict[token], idx] = 1
        return one_hot

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        x = self.one_hot_encoder(row.sequence)
        return x, np.array(row.label, dtype=self.y_type)

    def __len__(self):
        return len(self.dataframe)