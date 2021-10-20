#!/usr/bin/bash

log_file="data_processing.log"
cd data_preparing 
python data_cleaning.py > $log_file
python create_survival_time_dataset.py > $log_file
python extract_match_run_descriptions.py > $log_file
python extract_match_run_data.py > $log_file
python extract_organ_offer_history.py > $log_file

cd -
cp data_preparing/Processed\ Data/Organ\ Offer\ History/* data/
