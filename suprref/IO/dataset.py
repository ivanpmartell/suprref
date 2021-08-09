import os
import sys
import pathlib
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from pyfaidx import Fasta
from itertools import islice
from torch.utils.data import Dataset

try:
    from ..helper import Helper
    from ..annotation import Annotation
except:
    from helper import Helper
    from annotation import Annotation

class IODataset(Dataset):
    """
    Args:
        helperObj (Helper): Helper object.
        create_dataset (bool): True to create the dataset or False to load it for use in machine learning.
        y_type (np.dtype): Numpy data type used for the label. Difference for BCE and CE loss.
        get_pandas_dataframe (bool): True to create a pandas dataframe output.
    """
    LABEL_DICT = {'Non-Promoter': 0, 'Promoter': 1}
    LABEL_INT_DICT = {}
    LABEL_IO_DICT = {}
    LABEL_IO_INT_DICT = {}
    _helper: Helper
    _sld_file: str
    _num_sequences = 0
    _y_type: np.dtype

    def __init__(self, helperObj, create_dataset=False, y_type=np.int64, get_pandas_dataframe=False):
        self._y_type = y_type
        self._helper = helperObj
        self.LABEL_IO_DICT[self.LABEL_DICT['Promoter']] = 'I'
        self.LABEL_IO_DICT[self.LABEL_DICT['Non-Promoter']] = 'O'
        self.LABEL_INT_DICT[self.LABEL_DICT['Promoter']] = 'Promoter'
        self.LABEL_INT_DICT[self.LABEL_DICT['Non-Promoter']] = 'Non-Promoter'
        self.LABEL_IO_INT_DICT[self.LABEL_IO_DICT[self.LABEL_DICT['Promoter']]] = self.LABEL_DICT['Promoter']
        self.LABEL_IO_INT_DICT[self.LABEL_IO_DICT[self.LABEL_DICT['Non-Promoter']]] = self.LABEL_DICT['Non-Promoter']
        output_folder = self._helper.CONF_DICT[Helper._DICTKEY_EXPERIMENT_FOLDER]
        self._sld_file = os.path.join(output_folder, 'dataset.sld')
        if (create_dataset):
            if (get_pandas_dataframe):
                self.pandas_dataframe_creation()
            else:
                self.sld_creation(self._sld_file)
                self.sld_index_creation(self._sld_file)
        else:
            self.sld_index_creation(self._sld_file)
            self.sld_load()

    def sld_load(self):
        with open(self._sld_file+'i') as sldi_file:
            count = 0
            index = {}
            for line in sldi_file:
                record, start, end = line.rstrip().split('\t')
                index[record] = [int(start), int(end)]
                count += index[record][1] - index[record][0] +1
            self._num_sequences = count
        input_file_lines = Helper.get_number_of_lines(self._sld_file)
        with open(self._sld_file) as sld_file:
            next(sld_file)
            self._sld_record_in_memory = []
            self._sld_start_in_memory = []
            self._sld_pos_strand_label_in_memory = []
            self._sld_neg_strand_label_in_memory = []
            record = ""
            for i in tqdm(range(input_file_lines-1)):
                line = next(sld_file)
                if line.startswith('>'):
                    record = line.rstrip()[1:]
                    continue
                start, end, pos_strand_label, neg_strand_label = line.rstrip().split('\t')
                int_start = int(start)
                int_end = int(end)
                self._sld_record_in_memory.append(record)
                self._sld_start_in_memory.append(int_start)
                self._sld_pos_strand_label_in_memory.append(bool(self.LABEL_IO_INT_DICT[pos_strand_label[1]]))
                self._sld_neg_strand_label_in_memory.append(bool(self.LABEL_IO_INT_DICT[neg_strand_label[1]]))
        self._fasta_data = Fasta(self._helper.CONF_DICT[Helper._DICTKEY_INPUT_FILE], sequence_always_upper=True)

    def sld_index_creation(self, file):
        input_file_lines = Helper.get_number_of_lines(file)
        index_file = file+'i'
        if Helper.file_exists(index_file):
            print("Found index (.sldi) file")
            return
        print("Creating index (.sldi) file")
        with open(index_file, 'a') as ifile:
            with open(file) as sld_file:
                prev_record_start = 0
                prev_record = ""
                for i in tqdm(range(input_file_lines)):
                    line = next(sld_file)
                    if line.startswith('>'):
                        if prev_record_start != 0:
                            self._helper.save_file(file=ifile, save_func=Helper.save_sldi_line, continuous=True, record=prev_record, record_start=prev_record_start, record_end=i - 1)
                        prev_record_start = i + 1
                        prev_record = line.rstrip()[1:]
                self._helper.save_file(file=ifile, save_func=Helper.save_sldi_line, continuous=True, record=prev_record, record_start=prev_record_start, record_end=i)

    def sld_creation(self, output_file):
        print("Creating sequence label data (.sld) dataset file")
        window_size = self._helper.CONF_DICT[Helper._DICTKEY_CONFIGURATION][Helper._DICTKEY_WINDOW_SIZE]
        stride = self._helper.CONF_DICT[Helper._DICTKEY_CONFIGURATION][Helper._DICTKEY_STRIDE]
        output_folder = self._helper.CONF_DICT[Helper._DICTKEY_EXPERIMENT_FOLDER]
        input_file = self._helper.CONF_DICT[Helper._DICTKEY_INPUT_FILE]
        input_file_lines = Helper.get_number_of_lines(input_file)
        annotations_file = self._helper.CONF_DICT[Helper._DICTKEY_ANNOTATIONS]
        if Helper.file_exists(output_file):
            raise FileExistsError('Sequence label data (.sld) file already exists in the output folder')
        annotation = Annotation(annotations_file, self._helper)

        with open(output_file, 'a') as out_file:
            self._helper.save_file(file=out_file, save_func=Helper.save_header, continuous=True, filename=input_file)
            with open(input_file) as in_file:
                seq = ""
                record = ""
                coordinate = 0
                for line_number in tqdm(range(input_file_lines)):
                    line = next(in_file)
                    if line.startswith(">"):
                        coordinate = 0
                        seq = ""
                        record, *_ = line.split(' ', maxsplit=1)
                        record_id = record[1:].rstrip()
                        try:
                            annotation._data[record_id]
                        except:
                            continue
                        self._helper.save_file(file=out_file, save_func=Helper.save_record, continuous=True, record=record_id)
                        continue
                    try:
                        annotation._data[record_id]
                    except:
                        continue
                    seq += line.rstrip()
                    while len(seq) >= window_size:
                        if 'N' in seq:
                            seq = seq[stride:]
                        else:
                            window = seq[:window_size]
                            seq = seq[stride:]
                            pos_strand_label = self.get_label(annotation, record_id, '+', coordinate)
                            neg_strand_label = self.get_label(annotation, record_id, '-', coordinate)
                            self._helper.save_file(file=out_file, save_func=Helper.save_sld_line, continuous=True,
                                                start=coordinate, end=coordinate+window_size-1,
                                                pos_strand_label=pos_strand_label, neg_strand_label=neg_strand_label)
                        coordinate += stride


    def pandas_dataframe_creation(self):
        window_size = self._helper.CONF_DICT[self._helper._DICTKEY_CONFIGURATION][self._helper._DICTKEY_WINDOW_SIZE]
        stride = self._helper.CONF_DICT[self._helper._DICTKEY_CONFIGURATION][self._helper._DICTKEY_STRIDE]
        output_folder = self._helper.CONF_DICT[self._helper._DICTKEY_EXPERIMENT_FOLDER]
        input_file = self._helper.CONF_DICT[self._helper._DICTKEY_INPUT_FILE]
        _, file_and_extension = os.path.split(input_file)
        annotations_file = self._helper.CONF_DICT[Helper._DICTKEY_ANNOTATIONS]
        output_file = os.path.join(output_folder, 'dataset_file_record_strand.tsv')
        if Helper.file_exists(output_file):
            raise FileExistsError('Pandas dataframe (.tsv) file already exists in the output folder')
        annotation = Annotation(annotations_file, self._helper)
        with open(input_file) as f:
            for record in SeqIO.parse(f, "fasta"):
                for strand in ['+', '-']:
                    data = []
                    record_id = record.id
                    seq = record.seq._data
                    windows = (len(seq) - window_size) // stride + 2
                    for idx in tqdm(range(windows)):
                        start = idx * stride
                        end = start + window_size
                        if(end == len(seq)):
                            break
                        if(end > len(seq)):
                            start = len(seq) - window_size
                            end = start + window_size
                        sequence = seq[start:end]
                        if(strand == '-'):
                            sequence = self._helper.reverse_complement(sequence)
                        label = self.get_label(annotation, record_id, strand, start)
                        data.append([sequence, label])
                    df = pd.DataFrame(data, columns=['sequence', 'label'])
                    current_output_file = output_file.replace("file", file_and_extension).replace("record", record_id).replace("strand", strand)
                    self._helper.save_file(file=current_output_file, save_func=Helper.save_dataframe, continuous=False, dataframe=df)
                    df = None

    def get_label(self, annotation, chromosome, strand, coordinate):
        error_type = self._helper.CONF_DICT[Helper._DICTKEY_CONFIGURATION][Helper._DICTKEY_ERROR_TYPE]
        error_margin = self._helper.CONF_DICT[self._helper._DICTKEY_CONFIGURATION][self._helper._DICTKEY_ERROR_MARGIN]

        if(error_type == Helper._IO_ERROR_TYPES[0]): #sequence-overlap
            window_size = self._helper.CONF_DICT[self._helper._DICTKEY_CONFIGURATION][self._helper._DICTKEY_WINDOW_SIZE]
            downstream_length = self._helper.CONF_DICT[Helper._DICTKEY_CONFIGURATION][Helper._DICTKEY_SEQ_DOWNSTREAM_LEN]
            upstream_length = self._helper.CONF_DICT[Helper._DICTKEY_CONFIGURATION][Helper._DICTKEY_SEQ_UPSTREAM_LEN]
            tss_position = coordinate
            if(strand == '+'):
                tss_position += upstream_length
            else:
                tss_position += downstream_length
            closest = annotation.get_closest_TSS(chromosome, strand, tss_position)
            if(strand == '+'):
                closest_coordinates = [closest - upstream_length, closest + downstream_length]
            else:
                closest_coordinates = [closest - downstream_length, closest + upstream_length]
            window_coordinates = [coordinate, coordinate + window_size]
            if(window_coordinates[1] >= closest_coordinates[0] + error_margin[0] and closest_coordinates[1] - error_margin[1] >= window_coordinates[0]): #overlapping
                return self.LABEL_IO_DICT[self.LABEL_DICT['Promoter']]
            else:
                return self.LABEL_IO_DICT[self.LABEL_DICT['Non-Promoter']]
        elif(error_type == Helper._IO_ERROR_TYPES[1]): #tss-proximity
            downstream_length = self._helper.CONF_DICT[Helper._DICTKEY_CONFIGURATION][Helper._DICTKEY_SEQ_DOWNSTREAM_LEN]
            upstream_length = self._helper.CONF_DICT[Helper._DICTKEY_CONFIGURATION][Helper._DICTKEY_SEQ_UPSTREAM_LEN]
            tss_position = coordinate
            if(strand == '+'):
                tss_position += upstream_length
            else:
                tss_position += downstream_length
            closest = annotation.get_closest_TSS(chromosome, strand, tss_position)
            if(abs(tss_position - closest) < error_margin[0]):
                return self.LABEL_IO_DICT[self.LABEL_DICT['Promoter']]
            else:
                return self.LABEL_IO_DICT[self.LABEL_DICT['Non-Promoter']]

    def get_y(self):
        y_array = []
        for i in tqdm(range(len(self._sld_record_in_memory))):
            y_array.append(self._sld_pos_strand_label_in_memory[i])
            y_array.append(self._sld_neg_strand_label_in_memory[i])
        return np.array(y_array, dtype=self._y_type)

    def __getitem__(self, idx):
        int_idx = idx // 2
        record = self._sld_record_in_memory[int_idx]
        start = self._sld_start_in_memory[int_idx]
        pos_strand_label = self._sld_pos_strand_label_in_memory[int_idx]
        neg_strand_label = self._sld_neg_strand_label_in_memory[int_idx]
        end = start + self._helper.CONF_DICT[Helper._DICTKEY_CONFIGURATION][Helper._DICTKEY_WINDOW_SIZE]
        if int_idx % 2 == 0:
            x = self._fasta_data[record][start:end]
            y = int(pos_strand_label)
        else:
            x = self._fasta_data[record][start:end].reverse
            y = int(neg_strand_label)
        return self._helper.one_hot_encoder(x.seq), np.array(y, dtype=self._y_type)

    def __len__(self):
        return self._num_sequences * 2