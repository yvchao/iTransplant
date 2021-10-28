import os

import pandas as pd

column_definition = "./Raw Data/Match Data/PTR.htm"
match_records = "./Raw Data/Match Data/PTR.DAT"

with open(column_definition, "rb") as html_table:
    tables = pd.read_html(html_table, index_col=0)
    df = tables[0]
    columns = df.LABEL.to_list()

with open(match_records, "rb") as dat_table:
    match_data = pd.read_table(
        dat_table, names=columns, na_values=[" ", "."], encoding="cp1252"
    )

match = match_data.dropna(subset=["DONOR_ID", "WLREG_AUDIT_ID_CODE", "OFFER_ACCEPT"])

# only keep useful columns
selected_columns = [
    "DONOR_ID",
    "WLREG_AUDIT_ID_CODE",
    "LISTING_CTR_CODE",
    "OFFER_ACCEPT",
    "ORGAN_PLACED",
    "HOST_MATCH",
    "OPO_ALLOC_AUDIT_ID",
    "CLASS_ID",
    "PRIME_OPO_REFUSAL_ID",
    "TXC_REFUSAL_CD",
    "MATCH_SUBMIT_DATE",
]
match = match.loc[:, selected_columns]

# drop bypassed offers
match = match.loc[match["OFFER_ACCEPT"] != "B"]

# drop the cases that organ offers are rejected but no refusal reason given
drop_mask = (match.OFFER_ACCEPT == "N") & match.PRIME_OPO_REFUSAL_ID.isna()
match = match.loc[~drop_mask]

# primary reason of refusal
selected = match[match.OFFER_ACCEPT == "N"]
print("primary reason of offer refusal - OPO")
print(selected.PRIME_OPO_REFUSAL_ID.value_counts().head(10) / len(selected))
print("primary reason of offer refusal - TXC")
print(
    selected.TXC_REFUSAL_CD.value_counts().head(10)
    / (len(selected) - selected.TXC_REFUSAL_CD.isna().sum())
)

# case 1: acceptance: Y or Z
mask1 = (match.OFFER_ACCEPT == "Y") | (match.OFFER_ACCEPT == "Z")

# keep data for other 12 cases
# case 2: Rejection - Donor age or quality: 830
# case 3: Rejection - Donor size/weight: 831
# case 4: Rejection - Organ-specific donor issue: 837
# case 5: Rejection - Distance to travel or ship: 824
# case 6: Rejection - Donor Quality: 921
# case 7: Rejection - Positive serological tests: 834
# case 8: Rejection - Organ anatomical damage or defect: 836
# case 9: Rejection - Donor Age: 922
# case 10: Rejection - Donor Size/Weight: 923
# case 11: Rejection - Donor ABO: 924
# case 12: Rejection - Liver: Abnormal Biopsy: 942
# case 13: Rejection - Donor ABO: 832
reasons_to_keep = [830, 837, 831, 832, 824, 921, 834, 836, 922, 923, 924, 942]
mask = mask1
for reason in reasons_to_keep:
    mask2 = (match.OFFER_ACCEPT == "N") & (match.PRIME_OPO_REFUSAL_ID == reason)
    mask = mask | mask2

match = match.loc[mask]

# drop the case that HOST_MATCH==0 to remove some duplicated cases
match = match.loc[match.HOST_MATCH != 0]

# Sorting transplant centers by the number of accepted offers
hist_num = match[match.OFFER_ACCEPT == "Y"].LISTING_CTR_CODE.value_counts()
hist_den = match.LISTING_CTR_CODE.value_counts()
acceptance_rate = (hist_num / hist_den)[hist_den > 10000].sort_values(ascending=False)
print(
    "ID code of centers with top-10 acceptance rate of organ offers (more than 1000 acceptance)"
)
print(acceptance_rate.head(10))

centers = acceptance_rate[acceptance_rate > 0.01].index.to_list()
columns = [
    "DONOR_ID",
    "WLREG_AUDIT_ID_CODE",
    "OFFER_ACCEPT",
    "TXC_REFUSAL_CD",
    "MATCH_SUBMIT_DATE",
    "CLASS_ID",
    "OPO_ALLOC_AUDIT_ID",
]

save_dir = "./Processed Data/Extracted Match Data"

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

for center in centers:
    selected = match[match.LISTING_CTR_CODE == center]
    selected = selected[columns]
    selected.to_hdf(f"{save_dir}/match_data.h5", key=f"CTR{center}", mode="a")
