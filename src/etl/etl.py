# import pandas as pd
# from sqlalchemy import create_engine, text

# # 1. Ler CSV
# df = pd.read_csv("datasets/lcs.csv")

# print("Linhas no CSV:", len(df))
# print("Colunas:", df.columns.tolist())

# # 2. Limpar nomes de colunas (remover espaços)
# df.columns = df.columns.str.strip()

# # 3. Encodificar GENDER
# df["GENDER"] = df["GENDER"].map({"M": 0, "F": 1})

# # 4. Feature Engineering
# df["SMOKING_AGE"] = df["SMOKING"] * df["AGE"]

# sintomas = [
#     "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC DISEASE",
#     "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING",
#     "COUGHING", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
# ]
# df["SYMPTOM_COUNT"] = df[sintomas].sum(axis=1)

# df["RISK_SCORE"] = (
#     df["SMOKING"] * 2 +
#     df["AGE"] / 10 +
#     df["CHRONIC DISEASE"] * 2 +
#     df["WHEEZING"] * 1.5 +
#     df["COUGHING"] * 1.5 +
#     df["SHORTNESS OF BREATH"] * 2
# )

# # 5. Encodificar LUNG_CANCER (target)
# df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"YES": 1, "NO": 0})

# print("\nFeatures após engenharia:", df.columns.tolist())
# print("Linhas após ETL:", len(df))

# # 6. Conectar ao PostgreSQL
# engine = create_engine("postgresql://leonardoxavier:123456@localhost/healthcare_db")

# # 7. Salvar no PostgreSQL
# with engine.connect() as conn:
#     conn.execute(text("DROP TABLE IF EXISTS lung_cancer"))
#     conn.commit()

# df.to_sql("lung_cancer", con=engine, if_exists="replace", index=False)

# print("\nDados salvos no PostgreSQL na tabela 'lung_cancer'")

# # 8. Ler do PostgreSQL para confirmar
# df_db = pd.read_sql("SELECT * FROM lung_cancer", con=engine)
# print("\nLinhas lidas do PostgreSQL:", len(df_db))
