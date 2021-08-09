import pandas as pd

chr_dict: dict = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
                  '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12,
                  '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18,
                  '19': 19, '20': 20, '21': 21, '22': 22, 'X': 23, 'Y': 24}

annotations_file = "data/blast/genome_database/annotations.out"
chromosome = 1
downstream_length = 50

annotations = pd.read_csv(annotations_file, sep='\t', names=["Id", "pident", "nident", "start", "end", "strand"])
annotations.drop_duplicates(subset='Id', keep='first', inplace=True)

sga_annotations = []
for _, row in annotations.iterrows():
    if row.strand == 'plus':
        strand = '+'
        tss = row.end - downstream_length
    else:
        strand = '-'
        tss = row.end + downstream_length
    sga_annotations.append(("NC_{:06d}.15".format(chromosome), "TSS", tss, strand, 1, row.Id))

new_annotations = pd.DataFrame(sga_annotations, columns=["Id", "Type", "Position", "Strand", "Chromosome", "Gene"])
new_annotations.to_csv("data/human_transition.sga",sep='\t',header=False, index=False)