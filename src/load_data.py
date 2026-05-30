import pandas as pd

# Carregar dados
df = pd.read_csv("datasets/lcs.csv")

# Mostrar informações básicas
print("=== Shape ===")
print(df.shape)

print("\n=== Primeiras linhas ===")
print(df.head())

print("\n=== Tipos de dados ===")
print(df.dtypes)

print("\n=== Valores únicos por coluna ===")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} valores únicos")

print("\n=== Ausências ===")
print(df.isnull().sum())

print("\n=== Distribuição do target ===")
print(df["LUNG_CANCER"].value_counts())
