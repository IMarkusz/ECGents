#!/usr/bin/env python3
import xmltodict
import os
import pandas as pd
from tqdm import tqdm

def union(dict1, dict2):
    common_keys = dict1.keys() & dict2.keys()
    if common_keys:
        raise ValueError(f"Common keys found: {common_keys}")
    return {**dict1, **dict2}


DIR_PATH = "/mnt/cluster_vsbud-dl1-005/hackathon/"
META_CSV_PATH = "/mnt/cluster_vsbud-dl1-005/hackathon_meta/emergency_ikem_2017_2023.csv"

OUT_CSV_PATH_JUST_EXTRACTED = "/mnt/cluster_vsbud-dl1-005/hackathon_meta/metadata_just_extracted.csv"
OUT_CSV_PATH_MERGED = "/mnt/cluster_vsbud-dl1-005/hackathon_meta/metadata.csv"
OUT_MD_PATH = "/mnt/cluster_vsbud-dl1-005/hackathon_meta/metadata.md"

records_metadata = []

for f in tqdm(os.listdir(DIR_PATH)):
    path = os.path.join(DIR_PATH, f)

    with open(path, "rb") as xml:
        dict_ecg = xmltodict.parse(xml.read().decode("ISO-8859-1"))


    # RestingECG measurment

    filename = path[path.rfind("/")+1:]

    dict_metadata = {"file": filename}

    dict_metadata = union(dict_ecg["RestingECG"]["RestingECGMeasurements"], dict_metadata)

    records_metadata.append(dict_metadata)


df_extracted = pd.DataFrame(records_metadata)

df_csv = pd.read_csv(META_CSV_PATH, sep=";")

df = df_extracted.merge(df_csv, how="inner", on="file")

df_extracted.to_csv(OUT_CSV_PATH_JUST_EXTRACTED)
df.to_csv(OUT_CSV_PATH_MERGED)

with open(OUT_MD_PATH, "w") as f:
    f.write(df.to_markdown())