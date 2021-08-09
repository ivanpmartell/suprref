cd blast
makeblastdb -in ../$1.fna -title "Whole genome from $1" -dbtype nucl -out $1
cd ..
blastn -task megablast -query blast/human_promoters.fa -db blast/$1 -outfmt "7 qseqid length pident nident sstart send sstrand gaps sseq" -num_threads 4 -word_size 50 -perc_identity 95 | pv | awk '!/^#/ {print}' > blasted/$1.out
for f in blast/*.nhr; do
    rm $f
done
for f in blast/*.nin; do
    rm $f
done
for f in blast/*.nsq; do
    rm $f
done
python ../filter-blasted.py -input blasted/$1.out -output blasted/filtered/$1.out
rm blasted/$1.out
mv $1.fna done/$1.fna