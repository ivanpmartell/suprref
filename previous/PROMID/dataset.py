import os
import pathlib
import numpy as np
from Bio import SeqIO, File
import pandas as pd
import random
import matplotlib.pyplot as plt
from skorch.dataset import Dataset
from sklearn.model_selection import train_test_split
from scipy.special import softmax
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
from itertools import chain

promoter_loc = (-200, +400) # From paper
margin_of_error = 500 # From paper

class PROMIDSlidingDataset(Dataset):
    seqs_file: str
    seqs_length: int
    promoter_tss_loc: int
    num_records: int
    index_list: list
    batch_size: int
    y_type: np.dtype = np.int64
    indexed_file: File._IndexedSeqFileDict
    dataframe: pd.DataFrame
    current_data: pd.DataFrame
    dna_dict: dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    lbl_dict: dict = {'Non-Promoter': 0, 'Promoter': 1}

    def __init__(self, file, promoter_tss_loc, binary):
        self.seqs_file = file
        self.promoter_tss_loc = promoter_tss_loc
        #TODO: Change to parallel as in OURS
        self.indexed_file = SeqIO.index(file, 'fasta')
        self.index_list = list(self.indexed_file)
        self.num_records = len(self.indexed_file)
        if(binary):
            self.y_type = np.float32
    
    def get_sliding_sequences(self, name, seq, tss_loc):
        sequence = seq[tss_loc+promoter_loc[0]:tss_loc+promoter_loc[1]]
        if 'N' in sequence:
            return []
        if tss_loc < self.promoter_tss_loc-margin_of_error or tss_loc > self.promoter_tss_loc+margin_of_error:
            label = self.lbl_dict['Non-Promoter']
        else:
            label = self.lbl_dict['Promoter']
        return [name, sequence, label]

    def one_hot_encoder(self, seq):
        one_hot = np.zeros((len(self.dna_dict), len(seq)), dtype=np.float32)
        for idx, token in enumerate(seq):
            one_hot[self.dna_dict[token], idx] = 1
        return one_hot

    def set_current_data(self, target_idx, rand=True):
        record = self.indexed_file[self.index_list[target_idx]]
        locations = []
        if rand is True:
            for _ in range(self.batch_size):
                locations.append(self.get_rand_nonpromoter_tss_loc(len(record.seq), self.promoter_tss_loc))
        else:
            start = 0-promoter_loc[0]
            end = len(record.seq)-promoter_loc[1]
            locations = list(range(start, end))
        r_name = record.description.split(' ')[1]
        r_seq = record.seq._data
        records = []
        for tss_loc in locations:
            records.append(self.get_sliding_sequences(r_name, r_seq, tss_loc))
        self.current_data = pd.DataFrame(records, columns=['name', 'sequence', 'label']).dropna()

    def get_rand_nonpromoter_tss_loc(self, seq_len, tss_loc):
        us_or_ds = random.randint(0, 1)
        if(us_or_ds):
            nonpromoter_tss_loc = random.randrange(0-promoter_loc[0], tss_loc-margin_of_error)
        else:
            nonpromoter_tss_loc = random.randrange(tss_loc+margin_of_error+1, seq_len-promoter_loc[1])
        return nonpromoter_tss_loc
    
    def get_y_true(self):
        return self.current_data.label.values
    
    def __getitem__(self, idx):
        row = self.current_data.iloc[idx]
        x = self.one_hot_encoder(row.sequence.upper())
        return x, np.array(row.label, dtype=self.y_type)
    
    def __len__(self):
        return len(self.current_data)


class PROMIDValidationDataset(Dataset):
    seqs_length: int
    y_type: np.dtype = np.int64
    dataframe: pd.DataFrame
    dna_dict: dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    lbl_dict: dict = {'Non-Promoter': 0, 'Promoter': 1}

    def __init__(self, df, binary):
        self.dataframe = df
        example = self.dataframe.iloc[0]
        self.seqs_length = len(example.sequence)
        if(binary):
            self.y_type = np.float32
    
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

class PROMIDDataset(Dataset):
    seqs_length: int
    y_type: np.dtype = np.int64
    dataframe: pd.DataFrame
    val_dataset: PROMIDValidationDataset
    dna_dict: dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    lbl_dict: dict = {'Non-Promoter': 0, 'Promoter': 1}
    sliding_dataset: PROMIDSlidingDataset
    train_indices: int

    def __init__(self, file, tss_loc, binary, save_df):
        if('csv' in pathlib.Path(file).suffix):
            self.load_dataframe(file)
        else:
            print("Preprocessing: Creating the dataset sequences")
            seqs, neg_seqs = self.load_file(file, tss_loc)
            self.seqs_length = len(seqs[0])
            df = pd.DataFrame(seqs, columns=['sequence'])
            df['label'] = self.lbl_dict['Promoter']
            neg_df = pd.DataFrame(neg_seqs, columns=['sequence'])
            neg_df['label'] = self.lbl_dict['Non-Promoter']
            train_dataframe = df.append(neg_df, ignore_index=True)
            train_val_dataframe, test_dataframe = train_test_split(train_dataframe, train_size=0.9, stratify=train_dataframe['label'])
            self.dataframe, val_dataframe = train_test_split(train_val_dataframe, train_size=0.9, stratify=train_val_dataframe['label'])
            self.train_indices = self.dataframe.index[self.dataframe.index.values <= df.index.values.max()]
            self.val_dataset = PROMIDValidationDataset(df, binary)
            self.sliding_dataset = PROMIDSlidingDataset(file, tss_loc, binary)
            self.save_dataframe('models/promid/test_dataframe.csv', test_dataframe)
        if(binary):
            self.y_type = np.float32
        if(save_df is not None):
            self.save_dataframe(f'models/{save_df}/dataframe.csv')

    def load_file(self, file, tss_loc):
        seqs = []
        neg_seqs = []
        with open(file, 'rU') as fasta_file:
            for record in tqdm(SeqIO.parse(fasta_file, 'fasta')):
                r_name = record.description[:-1]
                r_seq = record.seq._data
                promoter = r_seq[tss_loc+promoter_loc[0]:tss_loc+promoter_loc[1]]
                if 'N' in promoter:
                    print("Ignoring %s: Sequence contained unknown base" % r_name)
                    continue
                seqs.append(promoter)

                condition = True
                tries = 0
                while(condition):
                    tries += 1
                    nonpromoter_tss_loc = self.get_rand_nonpromoter_tss_loc(len(r_seq), tss_loc)
                    
                    nonpromoter = r_seq[nonpromoter_tss_loc+promoter_loc[0]:nonpromoter_tss_loc+promoter_loc[1]]
                    condition = 'N' in nonpromoter
                if tries > 1:
                    print("Tried %d times to get nonpromoter: %s" % (tries, r_name))
                neg_seqs.append(nonpromoter)
                
        return seqs, neg_seqs
    
    def get_rand_nonpromoter_tss_loc(self, seq_len, tss_loc):
        us_or_ds = random.randint(0, 1)
        if(us_or_ds):
            nonpromoter_tss_loc = random.randrange(0-promoter_loc[0], tss_loc-margin_of_error)
        else:
            nonpromoter_tss_loc = random.randrange(tss_loc+margin_of_error+1, seq_len-promoter_loc[1])
        return nonpromoter_tss_loc

    def load_dataframe(self, file):
        self.dataframe = pd.read_csv(file)
        example = self.dataframe.iloc[0]
        self.seqs_length = len(example.sequence)
    
    def save_dataframe(self, file, df=None):
        print('Saving dataframe to: %s' % file)
        if df is None:
            self.dataframe.to_csv(file, index=False)
        else:
            df.to_csv(file, index=False)
    
    def append_false_positive_seqs(self, net):
        appended_num_seqs = 0
        random_plot = random.choice(self.train_indices)
        self.sliding_dataset.batch_size = net.batch_size
        for i in tqdm(self.train_indices):
            if random_plot == i:
                rand = False
            else:
                rand = True
            self.sliding_dataset.set_current_data(i, rand=rand)
            y_pred = net.predict_proba(self.sliding_dataset)
            y_pred = softmax(y_pred, axis=1)
            y_true = self.sliding_dataset.get_y_true()
            nonpromoter_filter = (y_true*-1+1).astype(bool)
            predictions = y_pred[nonpromoter_filter]
            if random_plot == i:
                self.plot_predictions(predictions, i, net.history[-1]['epoch'])
                continue
            filtered_sequences = self.sliding_dataset.current_data[nonpromoter_filter]
            top_false_positives_filter = predictions[:,self.lbl_dict['Promoter']]>0.95
            false_positives = filtered_sequences[top_false_positives_filter]
            #filtered_predictions = predictions[top_false_positives_filter]
            appended_num_seqs += len(false_positives)
            fp_df = pd.DataFrame(false_positives, columns=['sequence'])
            fp_df['label'] = self.lbl_dict['Non-Promoter']
            self.dataframe = self.dataframe.append(fp_df, ignore_index=True)
        print("\nAppended %d sequences to the negative dataset" % appended_num_seqs)
    
    def get_class_amounts(self):
        promoter_count = len(self.dataframe[self.dataframe['label'] == self.lbl_dict['Promoter']])
        nonpromoter_count = len(self.dataframe[self.dataframe['label'] == self.lbl_dict['Non-Promoter']])
        total_count = len(self.dataframe)
        if(promoter_count + nonpromoter_count != total_count):
            raise Exception("Wrong count")
        return promoter_count, nonpromoter_count

    def plot_predictions(self, predictions, idx, epoch):
        plt.plot(predictions[:,self.lbl_dict['Promoter']], linewidth=.1)
        plt.savefig(os.path.join('models/promid/images', 'Epoch-%d_id-%d.png' % (epoch, idx)))
        plt.cla()
        plt.clf()
        plt.close()

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