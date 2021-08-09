mkdir cv_dups_results
python cross_validate/cnnprom.py --input datasets_dups/cnnprom_TATA/dataframe.csv --output cv_dups_results/cnnprom_TATA/ --pool_size 4 --num_channels 200
python cross_validate/cnnprom.py --input datasets_dups/cnnprom_nonTATA/dataframe.csv --output cv_dups_results/cnnprom_nonTATA/
python cross_validate/cnnprom.py --input datasets_dups/cnnprom_complete/dataframe.csv --output cv_dups_results/cnnprom_complete/

python cross_validate/icnn.py --input datasets_dups/icnn_complete/dataframe.csv --output cv_dups_results/icnn_complete/

python cross_validate/dprom.py --input datasets_dups/dprom_TATA/dataframe.csv --output cv_dups_results/dprom_TATA/
python cross_validate/dprom.py --input datasets_dups/dprom_nonTATA/dataframe.csv --output cv_dups_results/dprom_nonTATA/
python cross_validate/dprom.py --input datasets_dups/dprom_complete/dataframe.csv --output cv_dups_results/dprom_complete/