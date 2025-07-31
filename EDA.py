
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv(r"Obesity.csv")

# Renomear a coluna alvo se necessário
if 'Obesity' in df.columns and 'Obesity_level' not in df.columns:
    df.rename(columns={'Obesity': 'Obesity_level'}, inplace=True)

# 🔹 Primeiras linhas
print("🔹 Primeiras linhas do dataset:")
print(df.head())

# 🔹 Informações gerais
print("\n🔹 Informações do dataset:")
print(df.info())

# 🔹 Verificar valores ausentes
print("\n🔹 Valores ausentes por coluna:")
print(df.isnull().sum())

# 🔹 Estatísticas descritivas
print("\n🔹 Estatísticas descritivas:")
print(df.describe(include='all'))

# 🔹 Identificar colunas numéricas e categóricas
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns.drop('Obesity_level')

# 🔹 Histogramas das variáveis numéricas
df[numeric_cols].hist(bins=15, figsize=(15, 10))
plt.suptitle("Distribuição das variáveis numéricas")
plt.tight_layout()
plt.show()

# 🔹 Boxplots para detectar outliers
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot de {col}")
plt.tight_layout()
plt.show()

# 🔹 Matriz de correlação
plt.figure(figsize=(12, 10))
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlação entre Variáveis Numéricas")
plt.show()

# 🔹 Gráficos de barras para variáveis categóricas
plt.figure(figsize=(18, 20))
for i, col in enumerate(categorical_cols):
    plt.subplot(5, 3, i + 1)
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f"Distribuição de {col}")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 🔹 Boxplots das variáveis numéricas por nível de obesidade
plt.figure(figsize=(15, 12))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(data=df, x='Obesity_level', y=col)
    plt.title(f"{col} por Nível de Obesidade")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 🔹 Gráficos de barras para variáveis categóricas por nível de obesidade
plt.figure(figsize=(18, 20))
for i, col in enumerate(categorical_cols):
    plt.subplot(5, 3, i + 1)
    sns.countplot(data=df, x=col, hue='Obesity_level', order=df[col].value_counts().index)
    plt.title(f"{col} por Nível de Obesidade")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
