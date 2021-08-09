mkdir evaluation
mkdir evaluation/cnnprom
# CNNPROM trained model on DPROM dataset
python test/cnnprom.py --output evaluation/cnnprom/ours.csv --input models/ours/dataframe.csv
python analysis.py --output evaluation/cnnprom/ours_results/ --results evaluation/cnnprom/ours.csv

mkdir evaluation/dprom
# DPROM trained model on CNNPROM dataset
python test/dprom.py --output evaluation/dprom/ours.csv --input models/ours/dataframe.csv
python analysis.py --output evaluation/dprom/ours_results/ --results evaluation/dprom/ours.csv

mkdir evaluation/icnn
# ICNN trained model on CNNPROM dataset
python test/icnn.py --output evaluation/icnn/ours.csv --input models/ours/dataframe.csv
python analysis.py --output evaluation/icnn/ours_results/ --results evaluation/icnn/ours.csv