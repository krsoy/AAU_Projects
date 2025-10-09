import requests
import zipfile
from io import BytesIO
import pandas as pd

url = "https://survey.stackoverflow.co/datasets/stack-overflow-developer-survey-2024.zip"
resp = requests.get(url)
resp.raise_for_status()

with zipfile.ZipFile(BytesIO(resp.content)) as z:
    # show all file names
    names = z.namelist()
    print("Files in ZIP:", names)

    # Option: read all CSVs into dict
    dfs = {}
    for fname in names:
        if fname.lower().endswith(".csv"):
            with z.open(fname) as f:
                dfs[fname] = pd.read_csv(f)

    # Example: show head of each
    for fname, df in dfs.items():
        print("===", fname, "→", df.shape)
        print(df.head())

    # Option: combine them (if same columns)
    # combined = pd.concat(dfs.values(), ignore_index=True)
    # print("Combined shape:", combined.shape)
df = dfs['survey_results_public.csv']