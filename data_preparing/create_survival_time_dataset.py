import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from selected_features import (
    COLUMN_IS_CATEGORICAL,
    INITIAL_PATIENT_FEATURES,
    ORGAN_FEATURES,
    PATIENT_FEATURES,
)

transplant_data_dir = "./Processed Data/Cleaned Transplant Data"
liver_data = pd.read_csv(f"{transplant_data_dir}/liver_data_cleaned.csv")

EXTRA_COLUMNS = [
    "PTIME",
    "PSTATUS",
    "DONOR_ID",
    "WL_ID_CODE",
    "LISTING_CTR_CODE",
    "register_date",
    "waitlist_removal_date",
    "tx_date",
    "composite_deathdate",
    "waitlist_deathdate",
]
data = liver_data[PATIENT_FEATURES + ORGAN_FEATURES + EXTRA_COLUMNS].copy()

data.loc[
    :,
    [
        "register_date",
        "waitlist_removal_date",
        "tx_date",
        "composite_deathdate",
        "waitlist_deathdate",
    ],
] = liver_data[
    [
        "register_date",
        "waitlist_removal_date",
        "tx_date",
        "composite_deathdate",
        "waitlist_deathdate",
    ]
].apply(
    pd.to_datetime
)

# remove all transplant occured before 2005 or after 2015
data = data.loc[~(data["tx_date"].dt.year < 2005)]
data = data.loc[~(data["tx_date"].dt.year > 2015)]

# case 1: transplanted, dead, survival time recorded
post_tx_death_mask = (
    (~data["tx_date"].isna()) & (data["PSTATUS"] == 1) & (~data["PTIME"].isna())
)
# case 2: not transplanted, dead in wait list
waitlist_death_mask = (data["tx_date"].isna()) & (~data["waitlist_deathdate"].isna())

tx_data = data.loc[post_tx_death_mask]
waitlist_data = data.loc[waitlist_death_mask]

tx_data = tx_data.dropna(subset=PATIENT_FEATURES + ORGAN_FEATURES)
waitlist_data = waitlist_data.dropna(subset=PATIENT_FEATURES)

# survial_data=pd.concat([tx_data,waitlist_data], axis=0)
survial_data = tx_data

for k in COLUMN_IS_CATEGORICAL.keys():
    if k in survial_data.columns and COLUMN_IS_CATEGORICAL[k]:
        survial_data.loc[:, k] = survial_data[k].astype("int")
        survial_data.loc[:, k] = survial_data[k].astype("category")

X = survial_data[PATIENT_FEATURES]
O = survial_data[ORGAN_FEATURES]

y = pd.DataFrame(index=survial_data.index, columns=["survival_time"], dtype="float")

# Post-transplant lifetime
y.loc[post_tx_death_mask, "survival_time"] = data[post_tx_death_mask]["PTIME"]

# Waitlist lifetime
# y.loc[waitlist_death_mask,'survival_time']=(data[waitlist_death_mask].waitlist_deathdate-data[waitlist_death_mask].register_date).dt.days

X = pd.get_dummies(X)
O = pd.get_dummies(O)
# O.loc[y['transplanted']==0,:]=0

save_dir = "./Processed Data/Cleaned Transplant Data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

joblib.dump(survial_data.dtypes, f"{save_dir}/feature_category.joblib")

X.to_hdf(f"{save_dir}/transplant_data_with_survival_time.h5", key="Patient", mode="w")
O.to_hdf(f"{save_dir}/transplant_data_with_survival_time.h5", key="Organ", mode="a")
y.to_hdf(
    f"{save_dir}/transplant_data_with_survival_time.h5", key="SurvivalTime", mode="a"
)
