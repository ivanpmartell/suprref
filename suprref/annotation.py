from collections import OrderedDict
import mysql.connector as mariadb
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os

try:
    from .helper import Helper
    from . import helper
except:
    from helper import Helper
    import helper

class Annotation:
    _data = {}
    _aliases_file: str
    _alias_map: dict
    _helper : Helper

    def __init__(self, file, helperObj):
        self._helper = helperObj
        self._aliases_file = os.path.join(helper.DATA_FOLDER, 'SUPR_chromosome_aliases.tsv')
        self.download_all_accessions()
        self.create_alias_map()
        file_path = Path(file)
        if file_path.suffix == '.sga': #EPD sga file
            #TODO: self.validate_epfl_mga_epdnew_annotations(file)
            self.get_epfl_mga_epdnew_annotations(file)
        elif file_path.suffix == '.bed': #riken bed file
            #TODO: self.validate_riken_fantom5_cage_annotations(file)
            self.get_riken_fantom5_cage_annotations(file)

    def get_epfl_mga_epdnew_annotations(self, file):
        with open(file) as f:
            for line in f:
                split_line = line.split('\t')
                chromosome = self._alias_map[split_line[0]]
                position = int(split_line[2])
                strand = split_line[3]
                gene = split_line[5].rstrip()
                try:
                    self._data[chromosome][strand].append([position, gene])
                except:
                    try:
                        self._data[chromosome][strand] = [[position, gene]]
                    except:
                        self._data[chromosome] = {strand: [[position, gene]]}
    
    def get_riken_fantom5_cage_annotations(self, file):
        with open(file) as f:
            for line in f:
                split_line = line.split('\t')
                chromosome = split_line[0]
                position = int(split_line[6])
                strand = split_line[5]
                score = int(split_line[4])
                try:
                    self._data[chromosome][strand].append([position, score])
                except:
                    try:
                        self._data[chromosome][strand] = [[position, score]]
                    except:
                        self._data[chromosome] = {strand: [[position, score]]}

    def get_closest_TSS(self, chromosome, strand, target):
        current_sequence = self._data[chromosome][strand]
        if (target <= current_sequence[0][0]): 
            return current_sequence[0][0]
        if (target >= current_sequence[-1][0]): 
            return current_sequence[-1][0]
        i = 0; j = len(current_sequence); mid = 0
        while (i < j):  
            mid = (i + j) // 2
            if (current_sequence[mid][0] == target): 
                return current_sequence[mid][0]
            if (target < current_sequence[mid][0]) : 
                if (mid > 0 and target > current_sequence[mid - 1][0]): 
                    return self.getClosest(current_sequence[mid - 1][0], current_sequence[mid][0], target) 
                j = mid 
            else : 
                if (mid < len(current_sequence) - 1 and target < current_sequence[mid + 1][0]): 
                    return self.getClosest(current_sequence[mid][0], current_sequence[mid + 1][0], target) 
                i = mid + 1
        return current_sequence[mid][0]
  
    @staticmethod
    def getClosest(val1, val2, target): 
        if (target - val1 >= val2 - target): 
            return val2 
        else: 
            return val1 

    def download_all_accessions(self):
        if Helper.file_exists(self._aliases_file):
            return 1
        mariadb_connection = mariadb.connect(host='genome-mysql.soe.ucsc.edu', user='genomep', password='password', database='hg38')
        cursor = mariadb_connection.cursor()
        neg_seqs = []
        cursor.execute(r"SELECT DISTINCT SCHEMA_NAME AS `database` FROM information_schema.SCHEMATA WHERE SCHEMA_NAME NOT IN ('information_schema', 'performance_schema', 'mysql') ORDER BY SCHEMA_NAME;")
        table = cursor.fetchall()
        cursor.close()
        with open(self._aliases_file, 'a') as aliases_file:
            for db in tqdm(table):
                cursor = mariadb_connection.cursor()
                try:
                    cursor.execute(f"SELECT alias, chrom FROM `{db[0]}`" + r".`chromAlias` AS chrAlias WHERE (chrAlias.alias LIKE 'NC_%' AND chrAlias.source = 'refseq');")
                except:
                    continue
                table = cursor.fetchall()
                self._helper.save_file(file=aliases_file, save_func=Helper.save_aliases_table, continuous=True, aliases_table=table)
                cursor.close()
        mariadb_connection.close()

    def create_alias_map(self):
        self._alias_map = {}
        aliases = pd.read_csv(self._aliases_file, sep="\t", names=['alias', 'chromosome'])
        previous_length = len(aliases)
        duplicated_aliases = aliases[aliases.alias.duplicated()]
        aliases = aliases.drop(duplicated_aliases.index).sort_values('alias')
        current_length = len(aliases)
        for idx, row in aliases.iterrows():
            self._alias_map[row.alias] = row.chromosome
        if current_length != previous_length:
            print("The following duplicated aliases have been detected:")
            print(duplicated_aliases)
            print("You can consolidate the chromosome aliases file by overwritting it")
            self._helper.save_file(file=self._aliases_file, save_func= Helper.save_dataframe, dataframe=aliases)

    def validate_epfl_mga_epdnew_annotations(self, file):
        #TODO:Use a regex
        raise NotImplementedError()

    def validate_riken_fantom5_cage_annotations(self, file):
        #TODO:Use a regex
        raise NotImplementedError()