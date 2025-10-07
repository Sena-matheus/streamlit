import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

st.set_page_config(page_title="Predição Estratégica de Crimes", layout="wide")
st.title("Predição Estratégica de Crimes - DELIT")
st.markdown("Ferramenta de previsão para planejamento policial baseado em Machine Learning.")

# Carregar dataset e modelo
@st.cache_data
def carregar_dados():
    df = pd.read_csv("dataset_ocorrencias_delegacia_5.csv")
    df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'])
    df['dia_semana'] = df['data_ocorrencia'].dt.day_name().replace({
        'Monday':'Segunda','Tuesday':'Terça','Wednesday':'Quarta',
        'Thursday':'Quinta','Friday':'Sexta','Saturday':'Sábado','Sunday':'Domingo'
    })
    df['mes'] = df['data_ocorrencia'].dt.month
    df['ano'] = df['data_ocorrencia'].dt.year
    df['hora_dia'] = df['data_ocorrencia'].dt.hour
    return df

df = carregar_dados()

# Carregar modelo
modelo = joblib.load("modelo.pkl")
colunas = joblib.load("colunas.pkl")

# Seleção do horizonte de previsão
st.subheader("Escolha o horizonte de previsão")
horizonte = st.selectbox(
    "Horizonte de previsão",
    ("Amanhã", "Próxima Semana", "Próximo Mês", "Próximo Semestre")
)

hoje = datetime.now()
if horizonte == "Amanhã":
    inicio = hoje + timedelta(days=1)
    fim = inicio
elif horizonte == "Próxima Semana":
    inicio = hoje + timedelta(days=1)
    fim = hoje + timedelta(days=7)
elif horizonte == "Próximo Mês":
    inicio = hoje + pd.DateOffset(months=1)
    fim = inicio + pd.DateOffset(months=1)
elif horizonte == "Próximo Semestre":
    inicio = hoje + pd.DateOffset(months=1)
    fim = inicio + pd.DateOffset(months=6)

# Filtros interativos no corpo da página
st.subheader("Filtros Interativos")
col1, col2, col3 = st.columns(3)

with col1:
    bairros = df['bairro'].unique()
    bairro_selecionado = st.multiselect("Bairro", bairros, default=bairros)

with col2:
    dias_semana = df['dia_semana'].unique()
    dia_selecionado = st.multiselect("Dia da Semana", dias_semana, default=dias_semana)

with col3:
    horas = range(0,24)
    hora_selecionada = st.slider("Hora do Dia", 0, 23, (0,23))

# Filtrando o dataset histórico relevante
df_filtrado = df[
    (df['bairro'].isin(bairro_selecionado)) &
    (df['dia_semana'].isin(dia_selecionado)) &
    (df['hora_dia'].between(hora_selecionada[0], hora_selecionada[1]))
]

# Previsão média baseada nos últimos dados
st.subheader(f"Previsão de crimes para: {horizonte}")

if df_filtrado.empty:
    st.warning("Não há dados suficientes para gerar previsão com os filtros selecionados.")
else:
    entrada_media = {
        'bairro': df_filtrado['bairro'].mode()[0],
        'arma_utilizada': df_filtrado['arma_utilizada'].mode()[0],
        'quantidade_vitimas': int(df_filtrado['quantidade_vitimas'].mean()),
        'quantidade_suspeitos': int(df_filtrado['quantidade_suspeitos'].mean()),
        'sexo_suspeito': df_filtrado['sexo_suspeito'].mode()[0],
        'idade_suspeito': int(df_filtrado['idade_suspeito'].mean()),
        'dia_semana': df_filtrado['dia_semana'].mode()[0],
        'hora_dia': int(df_filtrado['hora_dia'].mean()),
        'latitude': df_filtrado['latitude'].mean(),
        'longitude': df_filtrado['longitude'].mean()
    }

    entrada_df = pd.DataFrame([entrada_media])
    entrada_df = pd.get_dummies(entrada_df)
    entrada_df = entrada_df.reindex(columns=colunas, fill_value=0)

    previsao = modelo.predict(entrada_df)[0]
    st.success(f"Tipo de crime mais provável: **{previsao}**")

# Gráficos interativos para o período
st.subheader("Gráficos de previsão estratégica")

# Top crimes
top_crimes = df_filtrado['tipo_crime'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10,4))
sns.barplot(x=top_crimes.values, y=top_crimes.index, palette="Reds_r", ax=ax)
ax.set_xlabel("Quantidade de Ocorrências")
ax.set_ylabel("Tipo de Crime")
st.pyplot(fig)
plt.clf()

# Tipo de arma
armas = df_filtrado['arma_utilizada'].value_counts()
fig, ax = plt.subplots(figsize=(10,4))
sns.barplot(x=armas.values, y=armas.index, palette="Blues_r", ax=ax)
ax.set_xlabel("Quantidade")
ax.set_ylabel("Arma Utilizada")
st.pyplot(fig)
plt.clf()

# Crimes por dia da semana
dias = df_filtrado['dia_semana'].value_counts()
fig, ax = plt.subplots(figsize=(10,4))
sns.barplot(x=dias.index, y=dias.values, palette="Greens_r", ax=ax)
ax.set_xlabel("Dia da Semana")
ax.set_ylabel("Quantidade de Ocorrências")
st.pyplot(fig)
plt.clf()

# Crimes por hora do dia
horarios = df_filtrado['hora_dia'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10,4))
sns.barplot(x=horarios.index, y=horarios.values, palette="Purples_r", ax=ax)
ax.set_xlabel("Hora do Dia")
ax.set_ylabel("Quantidade de Ocorrências")
st.pyplot(fig)
plt.clf()

# Mapa interativo
st.subheader("Localização das ocorrências")
st.map(df_filtrado[['latitude', 'longitude']])