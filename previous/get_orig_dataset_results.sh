mkdir cv_original_results
python cross_validate/cnnprom.py --input datasets_original/cnnprom_nonTATA/dataframe.csv --output cv_original_results/cnnprom_nonTATA/

mkdir tts_original_results
python train/cnnprom.py --input datasets_original/cnnprom_nonTATA/train.csv --output tts_original_results/cnnprom_nonTATA/
python test/cnnprom.py --input datasets_original/cnnprom_nonTATA/test.csv --output tts_original_results/cnnprom_nonTATA/results_cnnprom_nonTATA.csv --model tts_original_results/cnnprom_nonTATA/model.pt
python analysis.py --output tts_original_results/cnnprom_nonTATA/cnnprom_nonTATA_results/ --results tts_original_results/cnnprom_nonTATA/results_cnnprom_nonTATA.csv

mkdir cv_orig_clean_results
python cross_validate/cnnprom.py --input datasets_original_clean/cnnprom_nonTATA/dataframe.csv --output cv_orig_clean_results/cnnprom_nonTATA/

mkdir tts_orig_clean_results
python train/cnnprom.py --input datasets_original_clean/cnnprom_nonTATA/train.csv --output tts_orig_clean_results/cnnprom_nonTATA/
python test/cnnprom.py --input datasets_original_clean/cnnprom_nonTATA/test.csv --output tts_orig_clean_results/cnnprom_nonTATA/results_cnnprom_nonTATA.csv --model tts_original_results/cnnprom_nonTATA/model.pt
python analysis.py --output tts_orig_clean_results/cnnprom_nonTATA/cnnprom_nonTATA_results/ --results tts_orig_clean_results/cnnprom_nonTATA/results_cnnprom_nonTATA.csv