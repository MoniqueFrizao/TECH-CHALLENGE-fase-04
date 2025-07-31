import streamlit as st
import pandas as pd
import joblib

# === Configurações iniciais ===
st.set_page_config(page_title="Sistema de Predição de Obesidade", layout="wide")
st.title("🏥 Sistema de Apoio ao Diagnóstico de Obesidade")
st.markdown("Este sistema utiliza um modelo de aprendizado de máquina para prever o nível de obesidade com base em informações pessoais e hábitos de vida.")

# === Caminhos dos arquivos salvos ===

modelo_path = "modelo_obesidade_xgboost.pkl"
encoders_path = "encoders.pkl"
scaler_path = "scaler.pkl"
target_encoder_path = "target_encoder.pkl"


# === Carregar modelo e objetos de pré-processamento ===
try:
    modelo = joblib.load(modelo_path)
    encoders = joblib.load(encoders_path)
    scaler = joblib.load(scaler_path)
    target_encoder = joblib.load(target_encoder_path)
except Exception as e:
    st.error(f"Erro ao carregar os arquivos: {e}")
    st.stop()

# === Interface ===
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("👤 Gênero", ['Feminino', 'Masculino'])
    age = st.slider("🎂 Idade", 10, 100, 25)
    height = st.number_input("📏 Altura (em metros)", min_value=1.0, max_value=2.5, value=1.70)
    weight = st.number_input("⚖️ Peso (em kg)", min_value=30.0, max_value=300.0, value=70.0)
    family_history = st.selectbox("👪 Histórico familiar de obesidade?", ['Sim', 'Não'])
    favc = st.selectbox("🍔 Consome alimentos calóricos com frequência?", ['Sim', 'Não'])
    fcvc = st.slider("🥦 Frequência de consumo de vegetais", 1.0, 3.0, 2.0)
    ncp = st.slider("🍽️ Número de refeições principais por dia", 1.0, 4.0, 3.0)

with col2:
    caec = st.selectbox("🍪 Belisca entre as refeições?", ['Não', 'Às vezes', 'Frequentemente', 'Sempre'])
    smoke = st.selectbox("🚬 Fuma?", ['Sim', 'Não'])
    ch2o = st.slider("💧 Litros de água por dia", 1.0, 3.0, 2.0)
    scc = st.selectbox("📊 Controla as calorias ingeridas?", ['Sim', 'Não'])
    faf = st.slider("🏃‍♀️ Atividade física (horas/semana)", 0.0, 40.0, 1.0)
    tue = st.slider("📱 Tempo em dispositivos eletrônicos (horas/dia)", 0.0, 13.0, 1.0)
    calc = st.selectbox("🍷 Consumo de álcool", ['Não', 'Às vezes', 'Frequentemente', 'Sempre'])
    mtrans = st.selectbox("🚌 Meio de transporte mais utilizado", ['Transporte público', 'Caminhada', 'Automóvel', 'Motocicleta', 'Bicicleta'])

# === Criar DataFrame com os dados ===
dados_entrada = pd.DataFrame([{
    'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
    'family_history': family_history, 'FAVC': favc, 'FCVC': fcvc, 'NCP': ncp,
    'CAEC': caec, 'SMOKE': smoke, 'CH2O': ch2o, 'SCC': scc,
    'FAF': faf, 'TUE': tue, 'CALC': calc, 'MTRANS': mtrans
}])

# === Mapeamento de valores em português para inglês ===
traducao = {
    'Feminino': 'Female', 'Masculino': 'Male',
    'Sim': 'yes', 'Não': 'no',
    'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always',
    'Transporte público': 'Public_Transportation', 'Caminhada': 'Walking',
    'Automóvel': 'Automobile', 'Motocicleta': 'Motorbike', 'Bicicleta': 'Bike'
}

for coluna in dados_entrada.columns:
    if dados_entrada[coluna].dtype == object:
        dados_entrada[coluna] = dados_entrada[coluna].replace(traducao)

# === Aplicar encoders salvos ===
try:
    for coluna, encoder in encoders.items():
        dados_entrada[coluna] = encoder.transform(dados_entrada[coluna])
except Exception as e:
    st.error(f"Erro ao aplicar os encoders: {e}")
    st.stop()

# === Normalizar variáveis numéricas ===
colunas_numericas = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
dados_entrada[colunas_numericas] = scaler.transform(dados_entrada[colunas_numericas])

# === Previsão ===
if st.button("🔮 Prever Nível de Obesidade"):
    try:
        indice_previsto = modelo.predict(dados_entrada)[0]
        classe_prevista = target_encoder.inverse_transform([indice_previsto])[0]

        st.success(f"**✅ Nível de obesidade previsto:** {classe_prevista}")

        with st.expander("ℹ️ O que significa esse nível?"):
            explicacoes = {
                'Insufficient_Weight': "Peso abaixo do ideal. Pode indicar desnutrição ou outros problemas de saúde.",
                'Normal_Weight': "Peso considerado saudável para a altura e idade.",
                'Overweight_Level_I': "Leve sobrepeso. Atenção aos hábitos alimentares e atividade física.",
                'Overweight_Level_II': "Sobrepeso moderado. Risco aumentado de problemas metabólicos.",
                'Obesity_Type_I': "Obesidade leve. Recomendável acompanhamento médico.",
                'Obesity_Type_II': "Obesidade moderada. Pode exigir intervenção clínica.",
                'Obesity_Type_III': "Obesidade severa. Alto risco à saúde, exige tratamento especializado."
            }
            st.write(explicacoes.get(classe_prevista, "Classe não reconhecida."))
    except Exception as e:
        st.error(f"Erro na previsão: {e}")




