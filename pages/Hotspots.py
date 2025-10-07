import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px


# Configuração da página
st.set_page_config(page_title="Mapa de Hotspots", page_icon="🔥", layout="wide")

# Função para carregar os dados
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv("dataset_ocorrencias_delegacia_5.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("../dataset_ocorrencias_delegacia_5.csv")
        except FileNotFoundError:
            st.error("❌ Arquivo 'dataset_ocorrencias_delegacia_5.csv' não encontrado.")
            st.stop()
    return df

df = carregar_dados()
df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'], errors='coerce')

# FILTROS
st.title("🔥 Mapa de Hotspots de Ocorrências")
st.markdown("Use os filtros abaixo para explorar padrões de criminalidade por bairro, tipo de crime e data.")

col_f1, col_f2 = st.columns(2)

bairros = sorted(df['bairro'].dropna().unique().tolist())
tipos_crime = sorted(df['tipo_crime'].dropna().unique().tolist())

bairro_sel = col_f1.selectbox("🏙️ Selecione o Bairro", ["Todos"] + bairros)
crime_sel = col_f2.selectbox("🚔 Selecione o Tipo de Crime", ["Todos"] + tipos_crime)

st.markdown("")  # Espaçamento visual
todas_datas = st.checkbox("📅 Considerar todas as datas", value=True)

if not todas_datas:
    data_sel = st.date_input(
        "Selecione a Data",
        value=df['data_ocorrencia'].min(),
        min_value=df['data_ocorrencia'].min(),
        max_value=df['data_ocorrencia'].max()
    )
else:
    data_sel = None

# Aplicar filtros
df_filtrado = df.copy()
if bairro_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado['bairro'] == bairro_sel]
if crime_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado['tipo_crime'] == crime_sel]
if not todas_datas and data_sel is not None:
    df_filtrado = df_filtrado[df_filtrado['data_ocorrencia'].dt.date == data_sel]

if df_filtrado.empty:
    st.warning("⚠️ Nenhum dado encontrado para os filtros selecionados.")
    st.stop()

# ANÁLISE GRÁFICA (com bloco visual)
st.markdown(
    """
    <style>
    .bloco-analise {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="bloco-analise">', unsafe_allow_html=True)
st.subheader("📊 Análise Gráfica Interativa")

opcao_grafico = st.radio(
    "Selecione o tipo de gráfico:",
    ("Gráfico de Barras", "Gráfico de Pizza", "Gráfico de Linha"),
    horizontal=True
)

if opcao_grafico == "Gráfico de Barras":
    ocorrencias_crime = df_filtrado['tipo_crime'].value_counts().reset_index()
    ocorrencias_crime.columns = ['Tipo de Crime', 'Ocorrências']
    fig = px.bar(ocorrencias_crime, x='Tipo de Crime', y='Ocorrências', color='Tipo de Crime',
                 title="Ocorrências por Tipo de Crime", height=400)
    st.plotly_chart(fig, use_container_width=True)

elif opcao_grafico == "Gráfico de Pizza":
    ocorrencias_bairro = df_filtrado['bairro'].value_counts().reset_index()
    ocorrencias_bairro.columns = ['Bairro', 'Ocorrências']
    fig = px.pie(ocorrencias_bairro, values='Ocorrências', names='Bairro',
                 title="Distribuição de Crimes por Bairro", height=400)
    st.plotly_chart(fig, use_container_width=True)

elif opcao_grafico == "Gráfico de Linha":
    df_tempo = df_filtrado.groupby(df_filtrado['data_ocorrencia'].dt.date).size().reset_index(name='Ocorrências')
    fig = px.line(df_tempo, x='data_ocorrencia', y='Ocorrências', markers=True,
                  title="Tendência Temporal das Ocorrências", height=400)
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# MAPA DE CALOR (no final)
st.subheader("🌍 Mapa de Calor de Ocorrências")

if 'latitude' not in df_filtrado.columns or 'longitude' not in df_filtrado.columns:
    st.error("❌ O dataset precisa conter as colunas 'latitude' e 'longitude' para gerar o mapa.")
else:
    mapa = folium.Map(
        location=[df_filtrado['latitude'].mean(), df_filtrado['longitude'].mean()],
        zoom_start=13,
        tiles='CartoDB positron'
    )
    heat_data = df_filtrado[['latitude', 'longitude']].dropna().values.tolist()
    HeatMap(heat_data, radius=12, blur=18, min_opacity=0.4).add_to(mapa)

    st_folium(mapa, width=1150, height=550)