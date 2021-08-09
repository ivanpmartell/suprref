import os
import sys
import argparse
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC

###########################################
# Command line interface
this_dir = os.path.dirname(os.path.relpath(sys.argv[0]))
default_out = os.path.join(this_dir, "data/blasted/filtered/ash17_filtered.out")
default_in = os.path.join(this_dir, "data/blasted/ash17.out")

parser = argparse.ArgumentParser(description=r"This script will filter a blasted file with a threshold for number of identical nucleotides. It will also remove duplicate sequences")
parser.add_argument('-input', 
        type=str, 
        help='Input file. This is the blasted file obtained using BLAST\'s custom format',
        default=default_in)
parser.add_argument('-output', 
        type=str, 
        help='Output file. This is the filtered file that will be created',
        default=default_out)
parser.add_argument('-threshold', 
        type=int, 
        help='Threshold that filters out sequences with less identical nucleotides',
        default=950)
args = parser.parse_args()
###########################################

def find_duplicates(df):
    seen = set()
    seen_ids = []
    seen2 = set()
    seen2_ids = []
    for idx, item in df.iterrows():
        if item.id in seen:
            seen2.add(item.id)
            seen2_ids.append(idx)
        else:
            seen.add(item.id)
            seen_ids.append(idx)
    return list(seen_ids), list(seen2_ids)

df = pd.read_csv(args.input, sep='\t', names=["id", "alength", "pident", "nident", "locS", "locE", "strand", "gaps", "sseq"])
filtered_df = df[df.nident > args.threshold]
no_dups, dups = find_duplicates(filtered_df)
df_no_dups = filtered_df.loc[no_dups]

fasta_output = args.output + ".fasta"
dummy_df = pd.DataFrame()
dummy_df.to_csv(fasta_output, sep=' ', header=False, index=False)

records = []
for idx, row in df_no_dups.iterrows():
        record = SeqRecord(Seq(row.sseq,
                                IUPAC.ambiguous_dna),
                                id=row.id, name="{0}".format(row.id),
                                description="alength:{0}_pident:{1}_nident:{2}_locS:{3}_locE:{4}_strand:{5}_gaps:{6}".format(row.alength, row.pident,row.nident,row.locS,row.locE,row.strand,row.gaps))
        records.append(record)
with open(fasta_output, "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")

df_no_dups.drop('sseq', axis=1, inplace=True)
df_no_dups.to_csv(args.output, sep='\t', header=False, index=False)