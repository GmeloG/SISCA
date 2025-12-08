from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parents[1]
df_raw = pd.read_csv(base_dir / "data" / "TSLA.csv")

print("Colunas do CSV:")
print(df_raw.columns)
print()

# escolher coluna de preço disponível
if "close" in df_raw.columns:
    price_col = "close"
else:
    raise ValueError("Nenhuma coluna 'Close' ou 'Adj Close' encontrada no TSLA.csv")

print(f"Coluna de preço usada: {price_col}")
print("Max preço :", df_raw[price_col].max())
print()

print("Linha do max preço:")
print(df_raw.loc[df_raw[price_col].idxmax(), ["date", price_col]])
