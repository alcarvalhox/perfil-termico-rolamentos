import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px  # Importamos o Plotly Express
from io import BytesIO

st.set_page_config(page_title="Perfil Térmico de Rodas", layout="wide")
st.title("🔍 Análise de Perfil Térmico de Rodas")
st.write("✅ App iniciado com sucesso!")

cut_off = 0.27

# A função de cache agora carrega o modelo do arquivo local
@st.cache_data
def load_model():
    try:
        # Abre e desserializa o modelo a partir do arquivo local
        with open("modelo_p_t_5_smt", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Erro: O arquivo do modelo 'modelo_p_t_5_smt' não foi encontrado.")
        return None
    except Exception as e:
        st.error(f"❌ Erro ao carregar o modelo: {e}")
        return None

# Carregamento do arquivo e botão
uploaded_file = st.file_uploader("📤 Faça upload do arquivo Excel para análise", type=["xlsx"])

if uploaded_file is not None:
    st.info("📁 Arquivo carregado com sucesso. Clique em 'Fazer a Análise' para continuar.")
    
    # Criamos um botão para iniciar a análise
    analyze_button = st.button("Fazer a Análise")
    
    if analyze_button:
        try:
            bd1 = pd.read_excel(uploaded_file, engine='openpyxl')
            array = bd1.values
            X = array[:, 0:28]

            model = load_model()
            if model is None:
                st.stop()

            preds = model.predict(X)
            preds_prob = model.predict_proba(X)
            results = np.where(preds_prob[:, 1] > cut_off, 1, 0)
            
            predicao = pd.DataFrame(results, columns=['Resultado'])
            predicao['Resultado'] = predicao['Resultado'].apply(lambda x: "Verdadeiro" if x == 1 else "Falso")

            proba = pd.DataFrame(preds_prob, columns=['Falso(%)', 'Verdadeiro(%)'])
            proba['Falso(%)'] *= 100
            proba['Verdadeiro(%)'] *= 100

            teste = pd.DataFrame(X)
            df = pd.concat([teste, predicao, proba], axis=1)

            st.subheader("📊 Relatório Gerado")
            st.dataframe(df)

            # --- Adicionando Análises Gráficas com Plotly ---
            st.subheader("📈 Análises Gráficas")
            
            # Gráfico de barras da distribuição dos resultados
            fig_resultados = px.histogram(df, x='Resultado', color='Resultado',
                                        title='Distribuição dos Resultados (Verdadeiro vs Falso)')
            st.plotly_chart(fig_resultados)

            # Gráfico de histograma da probabilidade de "Verdadeiro"
            fig_prob_verdadeiro = px.histogram(df, x='Verdadeiro(%)',
                                               title='Distribuição da Probabilidade de "Verdadeiro"',
                                               marginal='box',  # Adiciona um boxplot para análise de outliers
                                               color_discrete_sequence=['#1f77b4'])
            fig_prob_verdadeiro.add_vline(x=cut_off * 100, line_dash="dash", line_color="red", 
                                          annotation_text=f"Cut-off: {cut_off*100}%", 
                                          annotation_position="top right")
            st.plotly_chart(fig_prob_verdadeiro)

            # Gráfico de dispersão da probabilidade de "Verdadeiro" vs "Falso"
            fig_scatter_prob = px.scatter(df, x='Verdadeiro(%)', y='Falso(%)', color='Resultado',
                                          title='Probabilidade de "Verdadeiro" vs "Falso"',
                                          hover_data=['Resultado'])
            st.plotly_chart(fig_scatter_prob)

            # --- Download do Relatório ---
            def convert_df_to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                return output.getvalue()

            excel_data = convert_df_to_excel(df)

            st.download_button(
                label="📥 Baixar Relatório em Excel",
                data=excel_data,
                file_name="relatorio_Teste_falso_alarme_Quatis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.success("✅ Relatório gerado com sucesso!")
        except Exception as e:
            st.error(f"❌ Ocorreu um erro durante a análise: {e}")
else:
    st.info("📁 Aguarde o upload do arquivo Excel para iniciar a análise.")
   
