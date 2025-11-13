import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Lade die Excel-Dateien
print("=" * 50)
print("DATENANALYSE - INDEX.xlsx und Portfolio.xlsx")
print("=" * 50)

# INDEX Daten laden
index_df = pd.read_excel('DataStorage/INDEX.xlsx')
print("\n### INDEX.xlsx ###")
print(f"Shape: {index_df.shape}")
print(f"\nColumns: {list(index_df.columns)}")
print(f"\nFirst 5 rows:")
print(index_df.head())
print(f"\nData Types:")
print(index_df.dtypes)
print(f"\nMissing Values:")
print(index_df.isnull().sum())
print(f"\nBasic Statistics:")
print(index_df.describe())

# Portfolio Daten laden
portfolio_df = pd.read_excel('DataStorage/Portfolio.xlsx')
print("\n\n### Portfolio.xlsx ###")
print(f"Shape: {portfolio_df.shape}")
print(f"\nColumns: {list(portfolio_df.columns)}")
print(f"\nFirst 5 rows:")
print(portfolio_df.head())
print(f"\nData Types:")
print(portfolio_df.dtypes)
print(f"\nMissing Values:")
print(portfolio_df.isnull().sum())
print(f"\nBasic Statistics:")
print(portfolio_df.describe())

# Speichere eine kurze Zusammenfassung
with open('data_summary.txt', 'w') as f:
    f.write("DATA SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"INDEX.xlsx: {index_df.shape[0]} rows, {index_df.shape[1]} columns\n")
    f.write(f"Columns: {', '.join(index_df.columns)}\n\n")
    f.write(f"Portfolio.xlsx: {portfolio_df.shape[0]} rows, {portfolio_df.shape[1]} columns\n")
    f.write(f"Columns: {', '.join(portfolio_df.columns)}\n")

print("\n" + "=" * 50)
print("Summary saved to data_summary.txt")
print("=" * 50)
