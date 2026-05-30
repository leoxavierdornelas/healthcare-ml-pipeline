import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib

# 1. Conectar ao PostgreSQL
engine = create_engine("postgresql://leonardoxavier:123456@localhost/healthcare_db")

# 2. Ler dados do PostgreSQL
df = pd.read_sql("SELECT * FROM lung_cancer", con=engine)

print("Dados carregados do PostgreSQL:")
print("Linhas:", len(df))
print("Colunas:", df.columns.tolist())

# 3. Separar features e target
features = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", 
    "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE", "ALLERGY", 
    "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", 
    "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN",
    "SMOKING_AGE", "SYMPTOM_COUNT", "RISK_SCORE"
]

X = df[features]
y = df["LUNG_CANCER"]

print("\nFeatures:", features)
print("Shape X:", X.shape)
print("Shape y:", y.shape)

# 4. Dividir treino/teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTreino:", len(X_train), "Teste:", len(X_test))

# 5. Treinar XGBoost
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
print("\nModelo treinado!")

# 6. Avaliar
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n=== Acurácia ===")
print(f"Acurácia: {accuracy:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["NO_CANCER", "CANCER"]))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# 7. Importância das features
feature_importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n=== Top 10 Features Mais Importantes ===")
print(feature_importance.head(10).to_string(index=False))

# 8. Salvar modelo
joblib.dump(model, "models/xgboost_lung_cancer.pkl")
print("\nModelo salvo em 'models/xgboost_lung_cancer.pkl'")
