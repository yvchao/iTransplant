#!/usr/bin/bash

cd data_preparing 
OUTPUT_DIR=Processed\ Data
mkdir -p $OUTPUT_DIR
log_file="data_processing.log"
python data_cleaning.py > $log_file
python create_survival_time_dataset.py > $log_file
python extract_match_run_descriptions.py > $log_file
python extract_match_run_data.py > $log_file
python extract_organ_offer_history.py > $log_file
python offer_data_selection.py > $log_file
mv $log_file $OUTPUT_DIR

cd -