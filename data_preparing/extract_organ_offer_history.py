import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from extract_match_run_descriptions import create_allocation_type_lookup
from selected_features import (
    COLUMN_IS_CATEGORICAL,
    INITIAL_PATIENT_FEATURES,
    ORGAN_FEATURES,
    PATIENT_FEATURES,
)

transplant_data_dir = "./Processed Data/Cleaned Transplant Data"
waitlist_data_dir = "./Raw Data/Waiting List History"
match_data_dir = "./Processed Data/Extracted Match Data"

feature_dtypes = joblib.load(f"{transplant_data_dir}/feature_category.joblib")


def set_types(df, dtypes):
    for column in df.columns:
        if column in dtypes:
            df.loc[:, column] = df[column].astype(dtypes[column])
    return df


donor_data = pd.read_csv(f"{transplant_data_dir}/donor_data_cleaned.csv")[
    ["DONOR_ID"] + ORGAN_FEATURES
]
patient_data = pd.read_csv(f"{transplant_data_dir}/patient_data_cleaned.csv")[
    ["WL_ID_CODE", "register_date"] + PATIENT_FEATURES + INITIAL_PATIENT_FEATURES
]

donor_data = set_types(donor_data, feature_dtypes)
patient_data = set_types(patient_data, feature_dtypes)

donor_data.loc[:, "DONOR_ID"] = donor_data.DONOR_ID.astype(int)
donor_data = donor_data.set_index("DONOR_ID")
donor_data = donor_data[~donor_data.index.duplicated(keep="first")]

patient_data.loc[:, "WL_ID_CODE"] = patient_data.WL_ID_CODE.astype(int)
patient_data = patient_data.set_index("WL_ID_CODE")
patient_data = patient_data[~patient_data.index.duplicated(keep="first")]

column_definition = f"{waitlist_data_dir}/LIVER_WLHISTORY_DATA.htm"
wl_data = f"{waitlist_data_dir}/LIVER_WLHISTORY_DATA.DAT"

with open(column_definition, "rb") as html_table:
    tables = pd.read_html(html_table, index_col=0)
    df = tables[0]
    columns = df.LABEL.to_list()

with open(wl_data, "rb") as dat_table:
    wl_history = pd.read_table(
        dat_table, names=columns, na_values=[" ", "."], encoding="cp1252"
    )

wl_records = wl_history.dropna(
    subset=["WL_ID_CODE", "WLREG_AUDIT_ID_CODE", "MELD_PELD_LAB_SCORE"]
)
used_columns = [
    "WL_ID_CODE",
    "MELD_PELD_LAB_SCORE",
    "BILIRUBIN",
    "INR",
    "SERUM_CREAT",
    "SERUM_SODIUM",
    "ALBUMIN",
    "DIALYSIS_PRIOR_WEEK",
    "ENCEPH",
    "ASCITES",
    "DONCRIT_MIN_AGE",
    "DONCRIT_MAX_AGE",
    "DONCRIT_MIN_WGT",
    "DONCRIT_MAX_WGT",
    "DONCRIT_MAX_BMI",
]

wl_records = wl_records[wl_records.WL_ID_CODE.isin(patient_data.index)]
wl_records = wl_records.set_index("WLREG_AUDIT_ID_CODE")

lookup_table = create_allocation_type_lookup(
    f"{match_data_dir}/classification_dict.joblib"
)

save_dir = "./Processed Data/Organ Offer History"

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)


def generate_offer_history(
    center_id_code, wl_records, save_dir, feature_dtypes=feature_dtypes
):
    offer_data = pd.read_hdf(
        f"{match_data_dir}/match_data.h5", key=center_id_code, mode="r"
    )

    # change some data type
    offer_data.loc[:, "DONOR_ID"] = offer_data.DONOR_ID.astype(int)
    offer_data.loc[:, "WLREG_AUDIT_ID_CODE"] = offer_data.WLREG_AUDIT_ID_CODE.astype(
        int
    )
    offer_data.loc[:, "MATCH_SUBMIT_DATE"] = pd.to_datetime(
        offer_data["MATCH_SUBMIT_DATE"]
    )

    # only consider the offers related to donors and patients in the selected dataset
    print(f"total records number:{len(offer_data)} in center {center_id_code}")
    offer_data = offer_data[offer_data.DONOR_ID.isin(donor_data.index)]
    print(f"records number after removal of unknown donors:{len(offer_data)}")
    offer_data = offer_data[offer_data.WLREG_AUDIT_ID_CODE.isin(wl_records.index)]
    print(f"records number after removal of unknown patients:{len(offer_data)}")

    # collect patient and organ data for considered organ offers
    selected_donor_data = donor_data.loc[offer_data.DONOR_ID].copy()
    selected_wl_data = wl_records.loc[
        offer_data.WLREG_AUDIT_ID_CODE, used_columns
    ].copy()
    selected_patient_data = patient_data.loc[selected_wl_data.WL_ID_CODE].copy()

    mask = selected_wl_data.DIALYSIS_PRIOR_WEEK == "Y"
    selected_wl_data.loc[mask, "DIALYSIS_PRIOR_WEEK"] = 1
    selected_wl_data.loc[~mask, "DIALYSIS_PRIOR_WEEK"] = 0

    selected_patient_data.loc[:, "register_date"] = pd.to_datetime(
        selected_patient_data["register_date"]
    )

    delta = (
        offer_data.MATCH_SUBMIT_DATE.values - selected_patient_data.register_date.values
    )
    selected_patient_data.loc[:, "AGE"] = selected_patient_data.INIT_AGE + delta.astype(
        "timedelta64[Y]"
    ).astype(int)

    fill_features = [
        "MELD_PELD_LAB_SCORE",
        "BILIRUBIN",
    ]

    replace_features = [
        ("INR_TX", "INR"),
        ("FINAL_SERUM_SODIUM", "SERUM_SODIUM"),
        ("ALBUMIN_TX", "ALBUMIN"),
        ("DIAL_TX", "DIALYSIS_PRIOR_WEEK"),
        ("ALBUMIN_TX", "ALBUMIN"),
        ("ASCITES_TX", "ASCITES"),
        ("TBILI_TX", "BILIRUBIN"),
    ]

    for f in fill_features:
        mask = ~selected_wl_data[f].isna()
        selected_patient_data.loc[mask.to_list(), f] = selected_wl_data.loc[
            mask, f
        ].values

    for f_, f in replace_features:
        mask = ~selected_wl_data[f].isna()
        selected_patient_data.loc[mask.to_list(), f_] = selected_wl_data.loc[
            mask, f
        ].values

    mask = (selected_patient_data.AGE >= 18).to_list()
    offer_data = offer_data.loc[mask]
    selected_patient_data = selected_patient_data.loc[mask]
    selected_donor_data = selected_donor_data.loc[mask]

    selected_donor_data.loc[:, "SHARE_TY"] = lookup_table(
        offer_data[["OPO_ALLOC_AUDIT_ID", "CLASS_ID"]].values
    )

    for column in ["SHARE_TY"]:
        selected_donor_data.loc[:, column] = selected_donor_data[column].astype(
            feature_dtypes[column]
        )

    O = selected_donor_data[ORGAN_FEATURES]
    O = pd.get_dummies(O)

    X = selected_patient_data[PATIENT_FEATURES]  # +['outcome']]
    X = pd.get_dummies(X)

    mask = (X.isna().sum(axis=1) == 0).to_numpy() & (
        O.isna().sum(axis=1) == 0
    ).to_numpy()
    print(
        f"center: {center_id_code}, valid records: {np.sum(mask)}, total records {len(mask)}, ratio: {np.sum(mask)/len(mask)*100:.2f}"
    )
    X = X.iloc[mask]
    O = O.iloc[mask]
    offer_data = offer_data.iloc[mask]

    offer_data.to_hdf(f"{save_dir}/{center_id_code}.h5", key="Offer", mode="w")
    X.to_hdf(f"{save_dir}/{center_id_code}.h5", key="Patient", mode="a")
    O.to_hdf(f"{save_dir}/{center_id_code}.h5", key="Organ", mode="a")


with pd.HDFStore(f"{match_data_dir}/match_data.h5") as hdf:
    listing_centers = hdf.keys()

for center in listing_centers:
    generate_offer_history(center, wl_records, save_dir)
