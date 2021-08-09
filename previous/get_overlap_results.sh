mkdir overlap_results
# CNNPROM RESULTS
python train/cnnprom.py --input datasets_dups/cnnprom_TATA/dataframe.csv --output overlap_results/cnnprom_TATA/ --pool_size 4 --num_channels 200
python test/cnnprom.py --input datasets_dups/cnnprom_TATA/dataframe.csv --output overlap_results/cnnprom_TATA/results_cnnprom_TATA.csv --model overlap_results/cnnprom_TATA/model.pt --pool_size 4 --num_channels 200
python analysis.py --output overlap_results/cnnprom_TATA/cnnprom_TATA_results/ --results overlap_results/cnnprom_TATA/results_cnnprom_TATA.csv
python test/cnnprom.py --input datasets_dups/dprom_TATA/dataframe.csv --output overlap_results/cnnprom_TATA/results_dprom_TATA.csv --model overlap_results/cnnprom_TATA/model.pt --pool_size 4 --num_channels 200
python analysis.py --output overlap_results/cnnprom_TATA/dprom_TATA_results/ --results overlap_results/cnnprom_TATA/results_dprom_TATA.csv

python train/cnnprom.py --input datasets_dups/cnnprom_nonTATA/dataframe.csv --output overlap_results_test/cnnprom_nonTATA/
python test/cnnprom.py --input datasets_dups/cnnprom_nonTATA/dataframe.csv --output overlap_results/cnnprom_nonTATA/results_cnnprom_nonTATA.csv --model overlap_results/cnnprom_nonTATA/model.pt
python analysis.py --output overlap_results/cnnprom_nonTATA/cnnprom_nonTATA_results/ --results overlap_results/cnnprom_nonTATA/results_cnnprom_nonTATA.csv
python test/cnnprom.py --input datasets_dups/dprom_nonTATA/dataframe.csv --output overlap_results/cnnprom_nonTATA/results_dprom_nonTATA.csv --model overlap_results/cnnprom_nonTATA/model.pt
python analysis.py --output overlap_results/cnnprom_nonTATA/dprom_nonTATA_results/ --results overlap_results/cnnprom_nonTATA/results_dprom_nonTATA.csv

python train/cnnprom.py --input datasets_dups/cnnprom_complete/dataframe.csv --output overlap_results/cnnprom_complete/
python test/cnnprom.py --input datasets_dups/cnnprom_complete/dataframe.csv --output overlap_results/cnnprom_complete/results_cnnprom_complete.csv --model overlap_results/cnnprom_complete/model.pt
python analysis.py --output overlap_results/cnnprom_complete/cnnprom_complete_results/ --results overlap_results/cnnprom_complete/results_cnnprom_complete.csv
python test/cnnprom.py --input datasets_dups/dprom_complete/dataframe.csv --output overlap_results/cnnprom_complete/results_dprom_complete.csv --model overlap_results/cnnprom_complete/model.pt
python analysis.py --output overlap_results/cnnprom_complete/dprom_complete_results/ --results overlap_results/cnnprom_complete/results_dprom_complete.csv
python test/cnnprom.py --input datasets_dups/icnn_complete/dataframe.csv --output overlap_results/cnnprom_complete/results_icnn_complete.csv --model overlap_results/cnnprom_complete/model.pt
python analysis.py --output overlap_results/cnnprom_complete/icnn_complete_results/ --results overlap_results/cnnprom_complete/results_icnn_complete.csv

# ICNN RESULTS
python train/icnn.py --input datasets_dups/icnn_complete/dataframe.csv --output overlap_results/icnn_complete/
python test/icnn.py --input datasets_dups/cnnprom_complete/dataframe.csv --output overlap_results/icnn_complete/results_cnnprom_complete.csv --model overlap_results/icnn_complete/model.pt
python analysis.py --output overlap_results/icnn_complete/cnnprom_complete_results/ --results overlap_results/icnn_complete/results_cnnprom_complete.csv
python test/icnn.py --input datasets_dups/dprom_complete/dataframe.csv --output overlap_results/icnn_complete/results_dprom_complete.csv --model overlap_results/icnn_complete/model.pt
python analysis.py --output overlap_results/icnn_complete/dprom_complete_results/ --results overlap_results/icnn_complete/results_dprom_complete.csv
python test/icnn.py --input datasets_dups/icnn_complete/dataframe.csv --output overlap_results/icnn_complete/results_icnn_complete.csv --model overlap_results/icnn_complete/model.pt
python analysis.py --output overlap_results/icnn_complete/icnn_complete_results/ --results overlap_results/icnn_complete/results_icnn_complete.csv

# DPROM RESULTS
python train/dprom.py --input datasets_dups/dprom_TATA/dataframe.csv --output overlap_results/dprom_TATA/
python test/dprom.py --input datasets_dups/cnnprom_TATA/dataframe.csv --output overlap_results/dprom_TATA/results_cnnprom_TATA.csv --model overlap_results/dprom_TATA/model.pt
python analysis.py --output overlap_results/dprom_TATA/cnnprom_TATA_results/ --results overlap_results/dprom_TATA/results_cnnprom_TATA.csv
python test/dprom.py --input datasets_dups/dprom_TATA/dataframe.csv --output overlap_results/dprom_TATA/results_dprom_TATA.csv --model overlap_results/dprom_TATA/model.pt
python analysis.py --output overlap_results/dprom_TATA/dprom_TATA_results/ --results overlap_results/dprom_TATA/results_dprom_TATA.csv

python train/dprom.py --input datasets_dups/dprom_nonTATA/dataframe.csv --output overlap_results/dprom_nonTATA/
python test/dprom.py --input datasets_dups/cnnprom_nonTATA/dataframe.csv --output overlap_results/dprom_nonTATA/results_cnnprom_nonTATA.csv --model overlap_results/dprom_nonTATA/model.pt
python analysis.py --output overlap_results/dprom_nonTATA/cnnprom_nonTATA_results/ --results overlap_results/dprom_nonTATA/results_cnnprom_nonTATA.csv
python test/dprom.py --input datasets_dups/dprom_nonTATA/dataframe.csv --output overlap_results/dprom_nonTATA/results_dprom_nonTATA.csv --model overlap_results/dprom_nonTATA/model.pt
python analysis.py --output overlap_results/dprom_nonTATA/dprom_nonTATA_results/ --results overlap_results/dprom_nonTATA/results_dprom_nonTATA.csv

python train/dprom.py --input datasets_dups/dprom_complete/dataframe.csv --output overlap_results/dprom_complete/
python test/dprom.py --input datasets_dups/cnnprom_complete/test.csv --output overlap_results/dprom_complete/results_cnnprom_complete.csv --model overlap_results/dprom_complete/model.pt
python analysis.py --output overlap_results/dprom_complete/cnnprom_complete_results/ --results overlap_results/dprom_complete/results_cnnprom_complete.csv
python test/dprom.py --input datasets_dups/dprom_complete/dataframe.csv --output overlap_results/dprom_complete/results_dprom_complete.csv --model overlap_results/dprom_complete/model.pt
python analysis.py --output overlap_results/dprom_complete/dprom_complete_results/ --results overlap_results/dprom_complete/results_dprom_complete.csv
python test/dprom.py --input datasets_dups/icnn_complete/dataframe.csv --output overlap_results/dprom_complete/results_icnn_complete.csv --model overlap_results/dprom_complete/model.pt
python analysis.py --output overlap_results/dprom_complete/icnn_complete_results/ --results overlap_results/dprom_complete/results_icnn_complete.csv
