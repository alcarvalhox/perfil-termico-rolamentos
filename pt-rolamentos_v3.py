import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from io import BytesIO

st.title("Análise de Perfil Térmico de Rodas")
st.write("✅ App iniciado com sucesso!")

uploaded_file = st.file_uploader("Faça upload do arquivo Excel para análise", type=["xlsx"])
cut_off = 0.22

@st.cache_data
def load_model_from_drive(drive_url):
    try:
        file_id = drive_url.split('/')[-2]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(download_url)
        response.raise_for_status()
        model = pickle.load(BytesIO(response.content))
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

modelo_url = 'https://drive.google.com/file/d/1-D5IJg2-zvow5Jqvr4vb80KbmEr4R4o3/view?usp=sharing'

if uploaded_file is not None:
    try:
        bd1 = pd.read_excel(uploaded_file, engine='openpyxl')
        array = bd1.values
        X = array[:, 0:28]

        model = load_model_from_drive(modelo_url)
        if model is None:
            st.stop()

        preds = model.predict(X)
        preds_prob = model.predict_proba(X)
        results = np.where(preds_prob[:, 1] > cut_off, 1, 0)
        predicao = pd.DataFrame(results, columns=['Resultado'])

        for x in range(len(preds)):
            predicao.at[x, 'Resultado'] = "Verdadeiro" if predicao.at[x, 'Resultado'] == 1 else "Falso"

        proba = pd.DataFrame(preds_prob, columns=['Falso(%)', 'Verdadeiro(%)'])
        proba['Falso(%)'] *= 100
        proba['Verdadeiro(%)'] *= 100

        teste = pd.DataFrame(X)
        df = pd.concat([teste, predicao, proba], axis=1)

        st.subheader("Relatório Gerado")
        st.dataframe(df)

        def convert_df_to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            return output.getvalue()

        excel_data = convert_df_to_excel(df)
        st.download_button(
            label="Baixar Relatório em Excel",
            data=excel_data,
            file_name="relatorio_Teste_falso_alarme_Quatis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("###### Relatório salvo na pasta de destino ######")
    except Exception as e:
        st.error(f"Ocorreu um erro durante a análise: {e}")
