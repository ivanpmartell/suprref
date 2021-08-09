import pathlib
import numpy as np
from Bio import SeqIO
import pandas as pd
import random
from skorch.dataset import Dataset
import mysql.connector as mariadb
from tqdm import tqdm

class CNNPROMDataset(Dataset):
    seqs_length: int
    y_type: np.dtype = np.int64
    dataframe: pd.DataFrame
    dna_dict: dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    lbl_dict: dict = {'Non-Promoter': 0, 'Promoter': 1}

    def __init__(self, file, neg_file, num_negatives=None, num_positives=None, binary=False, save_df=None, test_set=False, split_ratio=0.30, drop_dups=True, dataset_folder="datasets", original=False):
        if('csv' in pathlib.Path(file).suffix):
            self.load_dataframe(file)
        else:
            seqs = self.load_file(file)
            self.seqs_length = len(seqs[0])
            df = pd.DataFrame(seqs, columns=['sequence'])
            if(drop_dups):
                df.drop_duplicates(inplace=True)
            try:
                if num_positives is not None:
                    df = df.sample(n=num_positives)
            except ValueError:
                print("Could not sample file. Check dataset correctness after it is created")
            df['label'] = self.lbl_dict['Promoter']
            if(neg_file is not None):
                neg_seqs = self.load_file(neg_file)
                neg_seqs_length = len(neg_seqs[0])
                if(self.seqs_length != neg_seqs_length):
                    raise Exception(r"Promoter and Non-Promoter sequence lengths don't match")
            else:
                print("Preprocessing: Creating the negative sequences")
                neg_seqs = self.create_negative_seqs(num_negatives)
            neg_df = pd.DataFrame(neg_seqs, columns=['sequence'])
            if(drop_dups):
                neg_df.drop_duplicates(inplace=True)
            neg_df['label'] = self.lbl_dict['Non-Promoter']
            try:
                if num_negatives is not None:
                    neg_df = neg_df.sample(n=num_negatives)
            except ValueError:
                print("Could not sample file. Check dataset correctness after it is created")

            self.dataframe = df.append(neg_df, ignore_index=True)
        if(binary):
            self.y_type = np.float32
        if(test_set):
            from sklearn.utils import resample
            from sklearn.model_selection import train_test_split
            if(original):
                train, test = train_test_split(self.dataframe, test_size=split_ratio)
            else:
                dprom_train_df = pd.read_csv(f'{dataset_folder}/{save_df.replace("cnnprom", "dprom")}/train.csv')
                dprom_test_df = pd.read_csv(f'{dataset_folder}/{save_df.replace("cnnprom", "dprom")}/test.csv')

                train_promoters_df = dprom_train_df[dprom_train_df['label'] == 1]
                resampled_train_df = resample(train_promoters_df, n_samples=int(num_positives*(1-split_ratio)),
                                                    replace=False, random_state=0)
                train, test = train_test_split(neg_df, test_size=split_ratio)
                train = train.append(resampled_train_df, ignore_index=True)

                test_promoters_df = dprom_test_df[dprom_test_df['label'] == 1]
                resampled_test_df = resample(test_promoters_df, n_samples=int(num_positives*split_ratio),
                                                    replace=False, random_state=0)
                test = test.append(resampled_test_df, ignore_index=True)

            train.to_csv(f'{dataset_folder}/{save_df}/train.csv',index=False)
            test.to_csv(f'{dataset_folder}/{save_df}/test.csv',index=False)
        if(save_df is not None):
            self.save_dataframe(f'{dataset_folder}/{save_df}/dataframe.csv')

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
    
    def create_negative_seqs(self, num_negatives):
        num_negatives = num_negatives + 5000 #in case of duplicates
        mariadb_connection = mariadb.connect(host='genome-mysql.soe.ucsc.edu', user='genomep', password='password', database='hg38')
        cursor = mariadb_connection.cursor()
        neg_seqs = []
        cursor.execute(r"SELECT chrom, strand, txStart, txEnd, exonStarts, exonEnds, name2 FROM `refGene` AS rg WHERE (rg.name LIKE 'NM_%' AND rg.chrom NOT LIKE '%\_%' AND rg.exonCount > 1) ORDER BY RAND() LIMIT " + str(num_negatives))
        table = cursor.fetchall()
        mariadb_df = pd.DataFrame(table, columns=cursor.column_names)
        cursor.close()
        mariadb_connection.close()
        while(len(neg_seqs) < num_negatives):
            for c in tqdm(mariadb_df.chrom.unique()):
                chrom_file = 'data/human_chrs/%s.fa' % c
                mariadb_chrom_df = mariadb_df.loc[mariadb_df['chrom'] == c]
                with open(chrom_file, 'rU')as cf:
                    chrom_seq = SeqIO.read(cf, 'fasta')
                    for _, row in mariadb_chrom_df.iterrows():
                        try:
                            if row['strand'] == '-':
                                exon1_end = int(row['exonEnds'].decode("utf-8").split(',')[-2]) - 1
                                gene_end = row['txStart'] + self.seqs_length
                                seq = 'N'
                                while 'N' in seq:
                                    random_neg = random.randint(gene_end, exon1_end)
                                    seq = chrom_seq.seq[random_neg - self.seqs_length: random_neg]
                                    seq = seq.upper()
                                neg_seq = self.create_antisense_strand(seq)
                            else:
                                exon1_end = int(row['exonEnds'].decode("utf-8").split(',')[0]) + 1
                                gene_end = row['txEnd'] - self.seqs_length
                                seq = 'N'
                                while 'N' in seq:
                                    random_neg = random.randint(exon1_end, gene_end)
                                    seq = chrom_seq.seq[random_neg: random_neg + self.seqs_length]
                                    seq = seq._data.upper()
                                neg_seq = seq
                            neg_seqs.append(neg_seq)
                            if(len(neg_seqs) >= num_negatives):
                                break
                        except Exception as e:
                            print('Error processing %s: %s' % (row['name2'], str(e)))
                if(len(neg_seqs) >= num_negatives):
                    break
            mariadb_df = mariadb_df.sample(frac=1).reset_index(drop=True)
        return neg_seqs

    reversal = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    def create_antisense_strand(self, org_seq):
        negs = []
        for letter in org_seq[::-1]:
            negs.append(self.reversal[letter])
        result = ''.join(negs)
        return result

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
