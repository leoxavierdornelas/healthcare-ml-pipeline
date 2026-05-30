import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib

# 1. Carregar dados
df = pd.read_csv('datasets/raw/lcs.csv')
df.columns = df.columns.str.strip()

print("Dados carregados:")
print(f"Linhas: {len(df)}")
print(f"Colunas: {df.columns.tolist()}")

# 2. BINARIZAR TODAS AS FEATURES NUMÉRICAS (> 0 vira 1)
binary_cols = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
               'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
               'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
               'SWALLOWING DIFFICULTY', 'CHEST PAIN']

for col in binary_cols:
    df[col] = (df[col] > 0).astype(int)
    print(f"{col}: {df[col].unique()}")

# 3. Criar features derivadas
df['SMOKING_AGE'] = df.apply(
    lambda r: (r['AGE'] - 40) if r['SMOKING'] == 1 and r['AGE'] > 40 else 0, 
    axis=1
)

symptom_cols = binary_cols
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

# 4. Converter GENDER e target
df['GENDER'] = df['GENDER'].map({'M': 0, 'F': 1})
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'NO': 0, 'YES': 1})

# 5. Features e target
features = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", 
    "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE", "ALLERGY", 
    "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", 
    "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN",
    "SMOKING_AGE", "SYMPTOM_COUNT", "RISK_SCORE"
]

X = df[features]
y = df['LUNG_CANCER']

# 6. Treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Treinar
model = xgb.XGBClassifier(
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
print("\nModelo treinado!")

# 8. Avaliar
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=["NO", "YES"]))

# 9. Salvar
joblib.dump(model, 'models/xgboost_lung_cancer.pkl')
print("\nModelo salvo!")
