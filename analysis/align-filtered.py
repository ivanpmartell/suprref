import os
import sys
import random
import argparse
import subprocess
import pandas as pd
from Bio import SeqIO, AlignIO, motifs
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC, Gapped

###########################################
# Command line interface
this_dir = os.path.dirname(os.path.relpath(sys.argv[0]))
default_out = os.path.join(this_dir, "data/blasted/aligned/")
default_in = os.path.join(this_dir, "data/blasted/filtered/")
default_p = os.path.join(this_dir, "data/blast/human_promoters.fa")

parser = argparse.ArgumentParser(description=r"This script will align each promoter sequence with its variants from other genomes")
parser.add_argument('-input', 
        type=str, 
        help='Input folder. This is the filtered folder containing BLAST\'s filtered output',
        default=default_in)
parser.add_argument('-promoters', 
        type=str, 
        help='Promoters file. This is the fasta file containing the promoters that have been used in the BLAST process',
        default=default_p)
parser.add_argument('-output', 
        type=str, 
        help='Output folder. This is the aligned folder that will contain MAFFT\'s aligned output',
        default=default_out)
args = parser.parse_args()
###########################################

def method1(promoter_list):
    alphabet = 'ACGTN'
    m = motifs.Motif(alphabet, motifs.Instances(promoter_list, alphabet))
    m.weblogo('testlogo.png')
    return m.pwm

promoter_dict = {}
with open(args.promoters, "r") as promoter_handle:
    for record in SeqIO.parse(promoter_handle, "fasta"):
        promoter_dict[record.id] = []

for f in os.listdir(args.input):
    if f.endswith(".fasta"):
        with open(os.path.join(args.input, f)) as fasta_handle:
            for record in SeqIO.parse(fasta_handle, "fasta"):
                promoter_dict[record.id].append((f, record.seq._data))

nonaligned = []
allsame = []
pwms = {}
temp_file = os.path.join(this_dir, "temp{0}.fa".format(random.randint(1,9999999)))
subprocess.call(['touch', temp_file])
for key, promoter_list in promoter_dict.items():
    records = []
    if len(promoter_list) < 2:
        nonaligned.append((key, promoter_list[0][0], promoter_list[0][1]))
        continue
    promoter_sequences = [p[1] for p in promoter_list]
    if len(set(promoter_sequences)) == 1:
        allsame.append((key, promoter_list[0][0], promoter_list[0][1]))
        continue

    for i, promoter in enumerate(promoter_list):
        record = SeqRecord(Seq(promoter[1], IUPAC.ambiguous_dna),
                            id="{0}.{1}".format(key, i), name="{0}".format(key),
                            description=promoter[0])
        records.append(record)
    with open(temp_file, "w") as temp_handle:
        SeqIO.write(records, temp_handle, "fasta")
    output_file = os.path.join(args.output, key + ".fa")
    mafft = 'mafft --auto {0} > {1}'.format(temp_file, output_file)
    mafft_process = subprocess.Popen(mafft,stdout=subprocess.DEVNULL, shell=True)
    mafft_process.wait()
    aligned_seqs = []
    with open(output_file, 'r') as out_handle:
        for record in SeqIO.parse(out_handle, "fasta"):
            aligned_seqs.append(record.seq._data)
    pwms[key] = method1([seq.upper().replace("-", "N") for seq in aligned_seqs])
subprocess.call(['rm', temp_file])

nonaligned_file = os.path.join(args.output, "nonaligned.out")
nonaligned_df = pd.DataFrame(nonaligned)
nonaligned_df.to_csv(nonaligned_file, sep='\t', header=False, index=False)

allsame_file = os.path.join(args.output, "allsame.out")
allsame_df = pd.DataFrame(allsame)
allsame_df.to_csv(allsame_file, sep='\t', header=False, index=False)