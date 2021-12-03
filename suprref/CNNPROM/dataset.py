import pathlib
import numpy as np
from Bio import SeqIO
import pandas as pd
import random
from skorch.dataset import Dataset
import mysql.connector as mariadb
from tqdm import tqdm

try:
    from ..helper import Helper
    from ..annotation import Annotation
except:
    from helper import Helper
    from annotation import Annotation

class CNNPROMDataset(Dataset):
    LABEL_DICT = {'Non-Promoter': 0, 'Promoter': 1}
    LABEL_INT_DICT = {}
    _y_type: np.dtype

    def __init__(self, helperObj, create_dataset=False, y_type=np.int64, get_pandas_dataframe=False):
        self.LABEL_INT_DICT[self.LABEL_DICT['Promoter']] = 'Promoter'
        self.LABEL_INT_DICT[self.LABEL_DICT['Non-Promoter']] = 'Non-Promoter'
        seqs = self.load_file(file)
        self.seqs_length = len(seqs[0])
        df = pd.DataFrame(seqs, columns=['sequence'])
        df['label'] = self.lbl_dict['Promoter']


        print("Preprocessing: Creating the negative sequences")
        neg_seqs = self.create_negative_seqs(num_negatives)
        neg_df = pd.DataFrame(neg_seqs, columns=['sequence'])
        neg_df['label'] = self.lbl_dict['Non-Promoter']
        self.dataframe = df.append(neg_df, ignore_index=True)

    def load_file(self, file):
        records = []
        with open(file, 'rU') as fasta_file:
            for record in SeqIO.parse(fasta_file, 'fasta'):
                r_seq = record.seq._data
                if 'N' not in r_seq:
                    records.append(r_seq)
        return records
    
    def create_negative_seqs(self, num_negatives):
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
                                exon1_end = int(row['exonEnds'].split(',')[-2]) - 1
                                gene_end = row['txStart'] + self.seqs_length
                                seq = 'N'
                                while 'N' in seq:
                                    random_neg = random.randint(gene_end, exon1_end)
                                    seq = chrom_seq.seq[random_neg - self.seqs_length: random_neg]
                                    seq = seq.upper()
                                neg_seq = self.create_antisense_strand(seq)
                            else:
                                exon1_end = int(row['exonEnds'].split(',')[0]) + 1
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
        int_idx = idx // 2
        record = self._sld_record_in_memory[int_idx]
        start = self._sld_start_in_memory[int_idx]
        pos_strand_label = self._sld_pos_strand_label_in_memory[int_idx]
        neg_strand_label = self._sld_neg_strand_label_in_memory[int_idx]
        window_size = self._helper.CONF_DICT[Helper._DICTKEY_CONFIGURATION][Helper._DICTKEY_SEQ_UPSTREAM_LEN] + self._helper.CONF_DICT[Helper._DICTKEY_CONFIGURATION][Helper._DICTKEY_SEQ_DOWNSTREAM_LEN]
        end = start + window_size
        if int_idx % 2 == 0:
            x = self._fasta_data[record][start:end]
            y = int(pos_strand_label)
        else:
            x = self._fasta_data[record][start:end].reverse
            y = int(neg_strand_label)
        return self._helper.one_hot_encoder(x.seq), np.array(y, dtype=self._y_type)
    
    def __len__(self):
        return len(self.dataframe)