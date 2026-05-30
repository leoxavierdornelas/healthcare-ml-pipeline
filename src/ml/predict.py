import pandas as pd
import joblib
import numpy as np

# Carregar modelo
model = joblib.load('models/xgboost_lung_cancer.pkl')

# Dados de exemplo (um paciente)
paciente = {
    'GENDER': ['M'],
    'AGE': [55],
    'SMOKING': [1],
    'YELLOW_FINGERS': [1],
    'ANXIETY': [0],
    'PEER_PRESSURE': [0],
    'CHRONIC DISEASE': [1],
    'FATIGUE': [1],
    'ALLERGY': [0],
    'WHEEZING': [1],
    'ALCOHOL CONSUMING': [0],
    'COUGHING': [1],
    'SHORTNESS OF BREATH': [1],
    'SWALLOWING DIFFICULTY': [1],
    'CHEST PAIN': [1]
}
df = pd.DataFrame(paciente)

# BINARIZAR TODAS AS FEATURES (0 ou 1 apenas)
numeric_cols = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
                'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
                'SWALLOWING DIFFICULTY', 'CHEST PAIN']

for col in numeric_cols:
    df[col] = (df[col] > 0).astype(int)  # Converter qualquer valor > 0 para 1

# Criar features derivadas
df['SMOKING_AGE'] = df.apply(
    lambda r: (r['AGE'] - 40) if r['SMOKING'] == 1 and r['AGE'] > 40 else 0, 
    axis=1
)

symptom_cols = [
    'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE',
    'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 
    'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'
]
df['SYMPTOM_COUNT'] = df[symptom_cols].sum(axis=1)

df['RISK_SCORE'] = df.apply(
    lambda r: (
        r['SMOKING'] * 2 +
        r['YELLOW_FINGERS'] * 1.5 +
        r['CHRONIC DISEASE'] * 1.5 +
        r['COUGHING'] * 2 +
        r['SHORTNESS OF BREATH'] * 2 +
        r['CHEST PAIN'] * 1.5 +
        r['WHEEZING'] * 1 +
        (1 if r['AGE'] > 50 else 0)
    ),
    axis=1
)

df['GENDER'] = df['GENDER'].map({'M': 0, 'F': 1})

features = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", 
    "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE", "ALLERGY", 
    "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", 
    "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN",
    "SMOKING_AGE", "SYMPTOM_COUNT", "RISK_SCORE"
]

X = df[features]

# Previsão
y_proba = model.predict_proba(X)[0]
prob_cancer = y_proba[1]  # Classe 1 = YES = CÂNCER
predicao = "CÂNCER DETECTADO" if prob_cancer > 0.5 else "SEM CÂNCER"

print(f"Probabilidade de câncer: {prob_cancer:.2%}")
print(f"Previsão: {predicao}")
