import warnings

import numpy as np
import pandas as pd


def read_offer_data(data_dir, center, yr_begin=2002, yr_end=2021):
    data_loc = f"{data_dir}/{center}.h5"
    offer_data = pd.read_hdf(data_loc, key="Offer", mode="r")[
        ["OFFER_ACCEPT", "MATCH_SUBMIT_DATE"]
    ]
    X = pd.read_hdf(data_loc, key="Patient", mode="r")
    O = pd.read_hdf(data_loc, key="Organ", mode="r")

    mask = (
        (offer_data.MATCH_SUBMIT_DATE.dt.year >= yr_begin)
        & (offer_data.MATCH_SUBMIT_DATE.dt.year < yr_end)
        & (offer_data.OFFER_ACCEPT != "Z")
    ).to_list()

    a = (offer_data.iloc[mask].OFFER_ACCEPT == "Y").values * 1.0
    X = X.iloc[mask]
    O = O.iloc[mask]

    data_source = {"patient": X, "organ": O, "action": a}
    return data_source


def aggregate_data(data_dir, centers):
    aggregate_data_source = {}
    for center in centers:
        data_source = read_offer_data(data_dir, center, yr_begin=2002, yr_end=2021)
        for k, v in data_source.items():
            if k in aggregate_data_source:
                aggregate_data_source[k].append(v)
            else:
                aggregate_data_source[k] = [v]

    for k, v in aggregate_data_source.items():
        if isinstance(v[0], pd.core.frame.DataFrame):
            aggregate_data_source[k] = pd.concat(v)
        elif isinstance(v[0], np.ndarray):
            aggregate_data_source[k] = np.concatenate(v)
        else:
            raise TypeError(f"cannot understand data type {type(v[0])}.")
    return aggregate_data_source


def detect_binary_columns(X):
    non_bin_columns = []
    bin_columns = []
    for i in range(X.shape[1]):
        unique_values = np.unique(X.iloc[:, i])
        if len(unique_values) > 2:
            non_bin_columns.append(i)
        elif len(unique_values) == 2:
            bin_columns.append(i)
        elif len(unique_values) == 1:
            bin_columns.append(i)
        else:
            raise ValueError(f"wrong data detected")
    return np.array(non_bin_columns), np.array(bin_columns)


def meld(X):
    creat = X.DIAL_TX_1 * 4.0 + X.DIAL_TX_0 * np.clip(X.CREAT_TX, 1, 4)
    MELD = np.round(
        9.57 * np.log(np.clip(creat, 1, 1e10))
        + 3.78 * np.log(np.clip(X.TBILI_TX, 1, 1e10))
        + 11.2 * np.log(np.clip(X.INR_TX, 1, 1e10))
        + 6.43
    )
    return MELD


def meld_na(X):
    MELD_i = meld(X)
    Na = np.clip(X.FINAL_SERUM_SODIUM, 125, 137)
    MELD_Na = MELD_i + 1.32 * (137 - Na) - 0.033 * MELD_i * (137 - Na)
    MELD = MELD_i * (MELD_i < 12) + MELD_Na * (MELD_i >= 12)
    return np.clip(MELD, 6, 40)


def create_criteria(X, O):
    df = pd.DataFrame()
    df.loc[:, "MELD"] = meld(X).values
    df.loc[:, "MELD-Na"] = meld_na(X).values
    df.loc[:, "Donor Age"] = O.AGE_DON.values
    df.loc[:, "Donor Height"] = O.HGT_CM_DON_CALC.values
    df.loc[:, "Donor Weight"] = O.WGT_KG_DON_CALC.values
    df.loc[:, "Donor AST"] = O.SGOT_DON.values
    df.loc[:, "Donor ALT"] = O.SGPT_DON.values
    df.loc[:, "Local Donor"] = O.SHARE_TY_3.values
    df.loc[:, "Regional Donor"] = O.SHARE_TY_4.values
    df.loc[:, "National Donor"] = O.SHARE_TY_5.values
    df.loc[:, "HCV Positive Donor"] = O.HEP_C_ANTI_DON_1.values
    df.loc[:, "HBV Positive Donor"] = O.HBV_CORE_DON_1.values
    df.loc[:, "African American Donor"] = O.ETHCAT_DON_2.values
    df.loc[:, "Female Donor"] = O.GENDER_DON_0.values
    df.loc[:, "Donation After Natural Death"] = O.deathcirc_1.values
    df.loc[:, "Non-heart-beating Donation"] = O.NON_HRT_DON_1.values
    df.loc[:, "Medium Degree MaS"] = O.macro_1.values
    df.loc[:, "High Degree MaS"] = O.macro_2.values
    df.loc[:, "Medium Degree MiS"] = O.micro_1.values
    df.loc[:, "High Degree MiS"] = O.micro_2.values
    return df


def create_criteria_discrete(X, O):
    df = pd.DataFrame()

    MELD = meld(X).values
    step = 5
    for i in range(0, 35, step):
        df.loc[:, f"{i:d}≤MELD<{i+step:d}"] = 1.0 * (MELD < (i + step)) * (MELD >= i)
    df.loc[:, f"MELD≥35"] = 1.0 * (MELD >= 35)

    MELD_Na = meld_na(X).values
    step = 5
    for i in range(0, 35, step):
        df.loc[:, f"{i:d}≤MELD-Na<{i+step:d}"] = (
            1.0 * (MELD_Na < (i + step)) * (MELD_Na >= i)
        )
    df.loc[:, f"MELD-Na≥35"] = 1.0 * (MELD_Na >= 35)

    age_don = O.AGE_DON.values
    step = 5
    df.loc[:, f"Donor Age<15"] = 1.0 * (age_don < 15)
    for i in range(15, 80, step):
        df.loc[:, f"{i:d}≤Donor Age<{i+step:d}"] = (
            1.0 * (age_don < (i + step)) * (age_don >= i)
        )
    df.loc[:, f"Donor Age≥80"] = 1.0 * (age_don >= 80)

    df.loc[:, "Donor Height"] = O.HGT_CM_DON_CALC.values
    df.loc[:, "Donor Weight"] = O.WGT_KG_DON_CALC.values

    AST = O.SGOT_DON.values
    AST_range = [100, 200, 500]
    for i in range(len(AST_range) - 1):
        df.loc[:, f"{AST_range[i]}≤Donor AST<{AST_range[i+1]}"] = (
            1.0 * (AST < AST_range[i + 1]) * (AST >= AST_range[i])
        )
    df.loc[:, f"≤Donor AST≥500"] = 1.0 * (AST >= 500)

    ALT = O.SGPT_DON.values
    ALT_range = [50, 100, 200, 500]
    for i in range(len(ALT_range) - 1):
        df.loc[:, f"{ALT_range[i]}≤Donor ALT<{ALT_range[i+1]}"] = (
            1.0 * (ALT < ALT_range[i + 1]) * (ALT >= ALT_range[i])
        )
    df.loc[:, f"ALT≥500"] = 1.0 * (ALT >= 500)

    df.loc[:, "Local Donor"] = O.SHARE_TY_3.values
    df.loc[:, "Regional Donor"] = O.SHARE_TY_4.values
    df.loc[:, "National Donor"] = O.SHARE_TY_5.values
    df.loc[:, "HCV Positive Donor"] = O.HEP_C_ANTI_DON_1.values
    df.loc[:, "HBV Positive Donor"] = O.HBV_CORE_DON_1.values
    df.loc[:, "African American Donor"] = O.ETHCAT_DON_2.values
    df.loc[:, "Female Donor"] = O.GENDER_DON_0.values
    df.loc[:, "Donation After Natural Death"] = O.deathcirc_1.values
    df.loc[:, "Non-heart-beating Donation"] = O.NON_HRT_DON_1.values
    df.loc[:, "Medium Degree MaS"] = O.macro_1.values
    df.loc[:, "High Degree MaS"] = O.macro_2.values
    df.loc[:, "Medium Degree MiS"] = O.micro_1.values
    df.loc[:, "High Degree MiS"] = O.micro_2.values
    return df


def create_dataset(data_source, discretize=False):
    patient_features = data_source["patient"]
    organ_features = data_source["organ"]
    if discretize:
        criteria = create_criteria_discrete(patient_features, organ_features)
    else:
        criteria = create_criteria(patient_features, organ_features)
    X = np.concatenate(
        [patient_features.values, organ_features.values, criteria.values], axis=1
    )
    y = data_source["action"]

    x_dim = patient_features.shape[1]
    o_dim = organ_features.shape[1]
    c_dim = criteria.shape[1]

    data_dict = {"X": {}, "y": {}}
    data_dict["X"]["patient"] = slice(0, x_dim)
    data_dict["X"]["organ"] = slice(x_dim, x_dim + o_dim)
    data_dict["X"]["criteria"] = slice(x_dim + o_dim, x_dim + o_dim + c_dim)

    data_dict["x_dim"] = x_dim
    data_dict["o_dim"] = o_dim
    data_dict["c_dim"] = c_dim

    data_dict["patient_features"] = patient_features.columns
    data_dict["organ_features"] = organ_features.columns
    data_dict["criteria"] = criteria.columns
    data_dict["y_labels"] = np.unique(y)

    column_dict = {}
    non_cat, cat = detect_binary_columns(patient_features)
    column_dict["patient"] = (non_cat, cat)
    non_cat, cat = detect_binary_columns(organ_features)
    column_dict["organ"] = (non_cat + x_dim, cat + x_dim)
    non_cat, cat = detect_binary_columns(criteria)
    column_dict["criteria"] = (non_cat + x_dim + o_dim, cat + x_dim + o_dim)
    return X, y, data_dict, column_dict
