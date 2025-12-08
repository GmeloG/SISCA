from __future__ import annotations

from pathlib import Path

import pandas as pd

from features import build_feature_dataframe


def main() -> None:
    
    input_path = Path(r"C:\Users\tomas\OneDrive\Documents\GitHub\SISCA/data/TSLA.csv")
    output_path = Path(r"C:\Users\tomas\OneDrive\Documents\GitHub\SISCA/data/TSLA_features.csv")


    #reading csv file (dataset)
    df_raw = pd.read_csv(input_path)

    print("Building features")
    df_feat = build_feature_dataframe(df_raw)

    #removing all the lines that sma200 doesn't exist 
    if "SMA_200" in df_feat.columns:
        df_feat = df_feat[df_feat["SMA_200"].notna()]

    #Save data in the folder 
    df_feat.to_csv(output_path, index=False)



if __name__ == "__main__":
    main()
