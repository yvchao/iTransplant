#!/usr/bin/bash

python data_cleaning.py
python create_survival_time_dataset.py
python extract_match_run_descriptions.py
python extract_match_run_data.py
python extract_organ_offer_history.py

cp ./Processed\ Data/Organ\ Offer\ History/* ../data/
