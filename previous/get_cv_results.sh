mkdir cv_results
python cross_validate/cnnprom.py --input datasets/cnnprom_TATA/dataframe.csv --output cv_results/cnnprom_TATA/ --pool_size 4 --num_channels 200
python cross_validate/cnnprom.py --input datasets/cnnprom_nonTATA/dataframe.csv --output cv_results/cnnprom_nonTATA/
python cross_validate/cnnprom.py --input datasets/cnnprom_complete/dataframe.csv --output cv_results/cnnprom_complete/

python cross_validate/icnn.py --input datasets/icnn_complete/dataframe.csv --output cv_results/icnn_complete/

python cross_validate/dprom.py --input datasets/dprom_TATA/dataframe.csv --output cv_results/dprom_TATA/
python cross_validate/dprom.py --input datasets/dprom_nonTATA/dataframe.csv --output cv_results/dprom_nonTATA/
python cross_validate/dprom.py --input datasets/dprom_complete/dataframe.csv --output cv_results/dprom_complete/