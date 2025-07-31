import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# Carregar o dataset
df = pd.read_csv(r"C:\Users\monique_sandoval\Downloads\Obesity.csv")

# Identificar colunas categóricas e numéricas
colunas_categoricas = df.select_dtypes(include='object').columns.tolist()
colunas_categoricas.remove('Obesity')  # variável alvo
colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Codificar variáveis categóricas
encoders = {}
for coluna in colunas_categoricas:
    le = LabelEncoder()
    df[coluna] = le.fit_transform(df[coluna])
    encoders[coluna] = le

# Codificar variável alvo
target_encoder = LabelEncoder()
df['Obesity'] = target_encoder.fit_transform(df['Obesity'])

# Normalizar variáveis numéricas
scaler = StandardScaler()
df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])

# Separar features e target
X = df.drop('Obesity', axis=1)
y = df['Obesity']

# Definir o modelo
modelo = RandomForestClassifier(random_state=42)

# Definir as métricas
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision_macro': make_scorer(precision_score, average='macro'),
    'recall_macro': make_scorer(recall_score, average='macro'),
    'f1_macro': make_scorer(f1_score, average='macro')
}

# Configurar validação cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Executar validação cruzada
resultados = cross_validate(modelo, X, y, cv=kf, scoring=scoring)

# Exibir resultados
print("📊 Resultados da Validação Cruzada (5-fold):")
print(f"Acurácia média: {resultados['test_accuracy'].mean():.4f}")
print(f"Precisão média (macro): {resultados['test_precision_macro'].mean():.4f}")
print(f"Recall médio (macro): {resultados['test_recall_macro'].mean():.4f}")
print(f"F1-score médio (macro): {resultados['test_f1_macro'].mean():.4f}")




