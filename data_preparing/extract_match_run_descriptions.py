import os

import joblib
import numpy as np
import pandas as pd

column_definition = "./Raw Data/Match Data/MATCH_CLASSIFICATIONS.htm"
match_classification = "./Raw Data/Match Data/MATCH_CLASSIFICATIONS.DAT"

with open(column_definition, "rb") as html_table:
    tables = pd.read_html(html_table, index_col=0)
    df = tables[0]
    columns = df.LABEL.to_list()

with open(match_classification, "rb") as dat_table:
    classification_data = pd.read_table(
        dat_table, names=columns, na_values=[" ", "."], encoding="cp1252"
    )

classification_dict = {}

drop_list = ["PRINT_FLG", "SHORT_DESCRIP", "CD", "ALLOC_AUDIT_ID"]
for audit_id in classification_data.ALLOC_AUDIT_ID.unique():
    classification_dict[audit_id] = (
        classification_data[classification_data.ALLOC_AUDIT_ID == audit_id]
        .drop(drop_list, axis=1)
        .set_index("ID")
    )

save_dir = "./Processed Data/Extracted Match Data"

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

joblib.dump(classification_dict, f"{save_dir}/classification_dict.joblib")


# SHARE_TYPE of organ: 3 - local, 4 - regional, 5 - national
def share_type(description):
    descrip = description.lower()
    if "opo" in descrip:
        return 3
    elif "local" in descrip:
        return 3
    elif "statewide" in descrip:
        return 3
    elif "150 nm" in descrip:
        return 3
    elif "region" in descrip:
        return 4
    elif "ropa" in descrip:
        return 4
    elif "250 nm" in descrip:
        return 4
    elif "nation" in descrip:
        return 5
    elif "500 nm" in descrip:
        return 5
    else:
        return np.nan


def create_allocation_type_lookup(
    dictionary_dir=f"{save_dir}/classification_dict.joblib",
):
    dictionary = joblib.load(dictionary_dir)

    def lookup_table(key_pairs):
        return [
            share_type(dictionary[key1].loc[key2].DESCRIP) for key1, key2 in key_pairs
        ]

    return lookup_table
