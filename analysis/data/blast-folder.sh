for f in ./*.fna; do
    tf=${f:2:${#f}-6}
    bash blast-genome.sh "$tf"
done