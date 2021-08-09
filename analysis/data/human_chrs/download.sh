for i in {1..22}
do
    wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr$i.fa.gz
done

for i in 'X' 'Y'
do
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr$i.fa.gz
done

gunzip *.gz