import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import io
import os

# --- Configurações do Streamlit ---
st.set_page_config(
    page_title="Análise de Perfil Térmico",
    layout="wide",
)

st.title("Análise de Perfil Térmico de Rodas")
st.markdown("---")

# --- Parâmetros ---
# ID do modelo no Google Drive
MODEL_ID = "1-D5IJg2-zvow5Jqvr4vb80KbmEr4R4o3"
MODEL_FILENAME = "modelo_p_t_4_smt.pkl" # O Streamlit espera um arquivo .pkl, não uma pasta

# --- Função para baixar o modelo ---
@st.cache_resource
def download_model():
    """Baixa o modelo do Google Drive e armazena em cache."""
    if not os.path.exists(MODEL_FILENAME):
        try:
            with st.spinner("Baixando o modelo de Machine Learning..."):
                gdown.download(id=MODEL_ID, output=MODEL_FILENAME, quiet=False)
            st.success("Modelo baixado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao baixar o modelo: {e}")
            return None
    
    with open(MODEL_FILENAME, 'rb') as f:
        model = pickle.load(f)
    return model

# --- Widgets da Interface ---
st.subheader("1. Carregar Dados de Análise")
uploaded_file = st.file_uploader("Carregue o arquivo XLSX com os dados de entrada", type=["xlsx"])

st.subheader("2. Configurações da Análise")
cut_off = st.slider(
    "Defina o valor de corte (cut_off) para a predição:",
    min_value=0.0,
    max_value=1.0,
    value=0.22,
    step=0.01
)

st.markdown("---")
# Botão para iniciar a análise
if st.button("Executar Análise", type="primary"):
    if uploaded_file is None:
        st.error("Por favor, carregue um arquivo para iniciar a análise.")
    else:
        st.subheader("3. Status da Execução")
        
        # Baixar e carregar o modelo
        model = download_model()
        if model is None:
            st.stop()
            
        try:
            with st.spinner("Lendo o arquivo de dados e executando a predição..."):
                # Ler o arquivo de entrada
                bd1 = pd.read_excel(uploaded_file)
                array = bd1.values
                X = array[:, 0:28]

                # Realizar a predição
                preds_prob = model.predict_proba(X)
                results = np.where(preds_prob[:, 1] > cut_off, 1, 0)
                
                # Criar o DataFrame de resultados
                predicao = pd.DataFrame(results, columns=['Resultado'])
                predicao['Resultado'] = predicao['Resultado'].map({1: "Verdadeiro", 0: "Falso"})
                
                proba = pd.DataFrame(preds_prob, columns=['Falso(%)', 'Verdadeiro(%)'])
                proba['Falso(%)'] = proba['Falso(%)'] * 100
                proba['Verdadeiro(%)'] = proba['Verdadeiro(%)'] * 100
                
                df = pd.concat([bd1, predicao, proba], axis=1)

            st.success("Análise concluída com sucesso! Visualização e Download disponíveis abaixo.")
            
            st.subheader("Prévia do Relatório Gerado")
            st.dataframe(df)

            # Preparar o arquivo para download
            with st.spinner("Gerando o arquivo Excel para download..."):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Resultados')
                output.seek(0)
            
            st.subheader("4. Download do Relatório")
            st.download_button(
                label="📥 Baixar Relatório XLSX",
                data=output,
                file_name="relatorio_perfil_termico.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            
        except Exception as e:
            st.error(f"Ocorreu um erro durante a análise: {e}")
            st.warning("Verifique se o arquivo de entrada está no formato correto (XLSX) e se possui as 28 colunas de dados esperadas.")
