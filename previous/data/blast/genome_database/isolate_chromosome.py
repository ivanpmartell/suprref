#Isolates a chromosome from an SGA file
import pandas as pd
from Bio import SeqIO

chromosome = 1
annotations_file = "data/human_complete.sga"
promoters_file = "data/human_complete.fa"

annotations = pd.read_csv(annotations_file, sep='\t', names=["Id", "Type", "Position", "Strand", "Chromosome", "Gene"])
annotations['Chromosome'] = annotations.Id.str[7:9].astype(int)

isolated_promoters = annotations[annotations['Chromosome'] == chromosome].Gene.tolist()

record_list = []
with open(promoters_file, 'r') as handle:
    for record in SeqIO.parse(handle, "fasta"):
        if (record.description.split(' ')[1] in isolated_promoters):
            record_list.append(record)

output_file = "data/blast/genome_database/human_promoters_chr{0}.fa".format(chromosome)
with open(output_file, "w+") as output_handle:
    SeqIO.write(record_list, output_handle, "fasta")
