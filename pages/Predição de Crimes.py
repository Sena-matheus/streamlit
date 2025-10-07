import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="Predição de Crimes", layout="wide")
st.title("Previsão de Crimes - DELIT")
st.markdown("Previsão do tipo de crime para o próximo mês usando Machine Learning.")

# Carregar dataset e modelo
@st.cache_data
def carregar_dados():
    df = pd.read_csv("dataset_ocorrencias_delegacia_5.csv")
    df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'])
    # Colunas derivadas
    df['dia_semana'] = df['data_ocorrencia'].dt.day_name().replace({
        'Monday':'Segunda','Tuesday':'Terça','Wednesday':'Quarta',
        'Thursday':'Quinta','Friday':'Sexta','Saturday':'Sábado','Sunday':'Domingo'
    })
    df['mes'] = df['data_ocorrencia'].dt.month
    df['ano'] = df['data_ocorrencia'].dt.year
    df['hora_dia'] = df['data_ocorrencia'].dt.hour
    return df

df = carregar_dados()

# Carregar modelo treinado e colunas
modelo = joblib.load("modelo.pkl")
colunas = joblib.load("colunas.pkl")

# Filtros do usuário
st.sidebar.header("Filtros")
anos = df['ano'].unique()
anos_selecionado = st.sidebar.multiselect("Ano", anos, default=anos)

meses = df['mes'].unique()
mes_selecionado = st.sidebar.multiselect("Mês", meses, default=meses)

bairros = df['bairro'].unique()
bairro_selecionado = st.sidebar.multiselect("Bairro", bairros, default=bairros)

df_filtrado = df[
    (df['ano'].isin(anos_selecionado)) &
    (df['mes'].isin(mes_selecionado)) &
    (df['bairro'].isin(bairro_selecionado))
]

# Previsão do próximo mês
st.subheader("🔹 Previsão do tipo de crime para o próximo mês")

# Próximo mês
proximo_mes = (datetime.now() + pd.DateOffset(months=1)).month

# Criar entrada média baseada nos últimos 3 meses
ultimos_dados = df[df['mes'].isin([proximo_mes-1, proximo_mes-2, proximo_mes-3])]

if ultimos_dados.empty:
    st.warning("Não há dados suficientes para gerar previsão para o próximo mês.")
else:
    entrada_media = {
        'bairro': ultimos_dados['bairro'].mode()[0],
        'arma_utilizada': ultimos_dados['arma_utilizada'].mode()[0],
        'quantidade_vitimas': int(ultimos_dados['quantidade_vitimas'].mean()),
        'quantidade_suspeitos': int(ultimos_dados['quantidade_suspeitos'].mean()),
        'sexo_suspeito': ultimos_dados['sexo_suspeito'].mode()[0],
        'idade_suspeito': int(ultimos_dados['idade_suspeito'].mean()),
        'dia_semana': ultimos_dados['dia_semana'].mode()[0],
        'hora_dia': int(ultimos_dados['hora_dia'].mean()),
        'latitude': ultimos_dados['latitude'].mean(),
        'longitude': ultimos_dados['longitude'].mean()
    }

    # Transformar em DataFrame e ajustar colunas
    entrada_df = pd.DataFrame([entrada_media])
    entrada_df = pd.get_dummies(entrada_df)
    entrada_df = entrada_df.reindex(columns=colunas, fill_value=0)

    previsao = modelo.predict(entrada_df)[0]
    st.success(f"Tipo de crime mais provável no próximo mês: **{previsao}**")