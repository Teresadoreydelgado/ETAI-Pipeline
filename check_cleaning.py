import sys

import yaml

from src.data import load_data
from src.preprocessing import clean_dataset

config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
config = yaml.safe_load(open(config_path))
df = clean_dataset(load_data(config["data"]["path"]), config["diagnostics"])

print(f"Missing values after cleaning ({config_path}):")
print(df.isna().sum()[df.isna().sum() > 0].to_string())

for col in config["preprocessing"]["categorical_features"]:
    if (df[col].astype(str).str.strip() == "?").any():
        print(f"WARNING: '{col}' still has a '?' category")