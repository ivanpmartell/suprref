from os import stat
from tqdm import tqdm
from pyfaidx import Fasta

try:
    from .helper import Helper
    from . import helper
except:
    from helper import Helper
    import helper

class Converter:
    _helper: Helper

    def __init__(self, helperObj):
        self._helper = helperObj

    def convert_sld2fasta(self):
        sld = self._helper.CONF_DICT[Helper._DICTKEY_INPUT_FILE]
        out = self._helper.CONF_DICT[Helper._DICTKEY_OUTPUT_FILE]
        input_file_lines = Helper.get_number_of_lines(sld)
        with open(sld, "r") as sld_file:
            data_file = next(sld_file).rstrip()
            if not data_file.startswith('@'):
                raise RuntimeError(msg="Wrong file format. No data file reference found.")
            data_file = data_file[1:]
            self._fasta_data = Fasta(data_file, sequence_always_upper=True)
        with open(out, "a") as out_file:
            with open(sld, "r") as sld_file:
                next(sld_file)
                record = ''
                for i in tqdm(range(input_file_lines-1)):
                    line = next(sld_file)
                    if line.startswith('>'):
                        record = line.rstrip()[1:]
                        continue
                    start, end, pos_strand_label, neg_strand_label = line.rstrip().split('\t')
                    int_start = int(start)
                    int_end = int(end) + 1
                    pos_x = self._fasta_data[record][int_start:int_end]
                    neg_x = -self._fasta_data[record][int_start:int_end]
                    self._helper.save_file(out_file, save_func=Helper.save_fasta_record, continuous=True, record=record, start=start, end=end, lbl=pos_strand_label, record_data=pos_x)
                    self._helper.save_file(out_file, save_func=Helper.save_fasta_record, continuous=True, record=record, start=start, end=end, lbl=neg_strand_label, record_data=neg_x)

    def convert_fasta2sld(self):
        #TODO
        raise NotImplementedError()

    def convert_sld2kmer(self):
        # This is for DNABERT
        sld = self._helper.CONF_DICT[Helper._DICTKEY_INPUT_FILE]
        out = self._helper.CONF_DICT[Helper._DICTKEY_OUTPUT_FILE]
        input_file_lines = Helper.get_number_of_lines(sld)
        with open(sld, "r") as sld_file:
            data_file = next(sld_file).rstrip()
            if not data_file.startswith('@'):
                raise RuntimeError(msg="Wrong file format. No data file reference found.")
            data_file = data_file[1:]
            self._fasta_data = Fasta(data_file, sequence_always_upper=True)
        with open(out, "a") as out_file:
            with open(sld, "r") as sld_file:
                next(sld_file)
                record = ''
                for i in tqdm(range(input_file_lines-1)):
                    line = next(sld_file)
                    if line.startswith('>'):
                        record = line.rstrip()[1:]
                        continue
                    start, end, pos_strand_label, neg_strand_label = line.rstrip().split('\t')
                    int_start = int(start)
                    int_end = int(end) + 1
                    pos_x = Converter.seq2kmer((self._fasta_data[record][int_start:int_end]).seq, 6)
                    neg_x = Converter.seq2kmer((-self._fasta_data[record][int_start:int_end]).seq, 6)
                    pos_lbl = Converter.change_label(pos_strand_label)
                    neg_lbl = Converter.change_label(neg_strand_label)
                    self._helper.save_file(out_file, save_func=Helper.save_tsv, continuous=True, seq=pos_x, label=pos_lbl)
                    self._helper.save_file(out_file, save_func=Helper.save_tsv, continuous=True, seq=neg_x, label=neg_lbl)

    @staticmethod
    def seq2kmer(seq, k):
        """
        FROM DNABERT https://github.com/jerryji1993/DNABERT/blob/master/motif/motif_utils.py
        """
        kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
        kmers = " ".join(kmer)
        return kmers

    @staticmethod
    def change_label(strand_label):
        """
        Change label depending on 3rd party program requirements
        O means Non-promoter (Out) and I means Promoter (In)
        """
        if strand_label[1:] == 'O':
            return 0
        elif strand_label[1:] == 'I':
            return 1
        return -1
    