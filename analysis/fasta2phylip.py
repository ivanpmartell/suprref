import os
import sys
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC

###########################################
# Command line interface
this_dir = os.path.dirname(os.path.relpath(sys.argv[0]))
default_out = os.path.join(this_dir, "data/blasted/aligned/FP012883.phy")
default_in = os.path.join(this_dir, "data/blasted/aligned/FP012883.fa")

parser = argparse.ArgumentParser(description=r"This script will convert an aligned file from fasta format to phylip format")
parser.add_argument('-input', 
        type=str, 
        help='Input file. This is the aligned file obtained from MAFFT',
        default=default_in)
parser.add_argument('-output', 
        type=str, 
        help='Output file. This is the corresponding phylip file',
        default=default_out)
args = parser.parse_args()
###########################################

records = []
with open(args.input, "r") as input_handle:
    for record in SeqIO.parse(input_handle, "fasta"):
        r = SeqRecord(Seq(record.seq._data,
                                IUPAC.ambiguous_dna),
                                id=record.id.replace('.',''), name="{0}".format(record.id),
                                description=record.description)
        records.append(r)

count = SeqIO.write(records, args.output, "phylip")
print("Converted %i records" % count)