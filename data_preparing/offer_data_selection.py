import os
import shutil

import numpy as np
import pandas as pd
import tqdm

data_dir = "Processed Data"
match_data_dir = os.path.join(data_dir, "Extracted Match Data")
offer_data_dir = os.path.join(data_dir, "Organ Offer History")
target_dir = "../data"


with pd.HDFStore(os.path.join(match_data_dir, "match_data.h5")) as hdf:
    listing_centers = hdf.keys()

columns = [
    "center id",
    "offer number",
    "acceptance rate",
    "accepted offers",
    "offer number (filtered)",
    "acceptance rate (filtered)",
    "accepted offers (filtered)",
]
stats = []
for center_id_code in listing_centers:
    offer_data = pd.read_hdf(
        os.path.join(match_data_dir, "match_data.h5"), key=center_id_code, mode="r"
    )
    offer_data = offer_data[offer_data.OFFER_ACCEPT != "Z"]
    acceptance_rate = (offer_data.OFFER_ACCEPT == "Y").sum() / len(offer_data)

    filtered_offer_data = pd.read_hdf(
        os.path.join(offer_data_dir, f"{center_id_code[1:]}.h5"), key="Offer", mode="r"
    )
    filtered_offer_data = filtered_offer_data[filtered_offer_data.OFFER_ACCEPT != "Z"]

    filtered_acceptance_rate = (filtered_offer_data.OFFER_ACCEPT == "Y").sum() / len(
        filtered_offer_data
    )

    stats.append(
        [
            center_id_code[1:],
            len(offer_data),
            acceptance_rate,
            (offer_data.OFFER_ACCEPT == "Y").sum(),
            len(filtered_offer_data),
            filtered_acceptance_rate,
            (filtered_offer_data.OFFER_ACCEPT == "Y").sum(),
        ]
    )

stats = pd.DataFrame(stats, columns=columns)
stats = stats.sort_values("accepted offers (filtered)", ascending=False).reset_index(
    drop=True
)
stats.to_csv(os.path.join(target_dir, "acceptancerate.csv"))

with tqdm.trange(10) as tbar:
    for i in tbar:
        file_name = f"{stats.iloc[i]['center id']}.h5"
        source_data_file = os.path.join(offer_data_dir, file_name)
        shutil.copy2(source_data_file, target_dir)
