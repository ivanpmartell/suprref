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
                    self._helper.save_file(out_file, save_func=Helper.save_fasta_record, continuous=True, ecord=record, start=start, end=end, lbl=neg_strand_label, record_data=neg_x)

    def convert_fasta2sld(self):
        #TODO
        raise NotImplementedError()
    