

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib

# 1. Carregar o dataset
df = pd.read_csv(r"C:\Users\monique_sandoval\Downloads\Obesity.csv")

# 2. Identificar colunas categóricas e numéricas
colunas_categoricas = df.select_dtypes(include='object').columns.tolist()
colunas_categoricas.remove('Obesity')  # variável alvo
colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# 3. Codificar variáveis categóricas
encoders = {}
for coluna in colunas_categoricas:
    le = LabelEncoder()
    df[coluna] = le.fit_transform(df[coluna])
    encoders[coluna] = le

# 4. Codificar variável alvo
target_encoder = LabelEncoder()
df['Obesity'] = target_encoder.fit_transform(df['Obesity'])

# 5. Normalizar variáveis numéricas
scaler = StandardScaler()
df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])

# 6. Separar features e target
X = df.drop('Obesity', axis=1)
y = df['Obesity']

# 7. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Treinar modelo XGBoost
modelo = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
modelo.fit(X_train, y_train)

# 9. Avaliação
y_pred = modelo.predict(X_test)
print("✅ Avaliação do modelo:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")

# 10. Salvar modelo e objetos de pré-processamento
joblib.dump(modelo, 'modelo_obesidade_xgboost.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')

print("✅ Modelo e objetos de pré-processamento salvos com sucesso.")
