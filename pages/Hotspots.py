import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
from datetime import datetime

# Configuração da página
st.set_page_config(page_title="Mapa de Hotspots", page_icon="", layout="wide")

# Função para carregar os dados
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv("dataset_ocorrencias_delegacia_5.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("../dataset_ocorrencias_delegacia_5.csv")
        except FileNotFoundError:
            st.error("Arquivo 'dataset_ocorrencias_delegacia_5.csv' não encontrado.")
            st.stop()
    
    df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'], errors='coerce')
    # Prepara coluna hora_dia para gráficos
    df['hora_dia'] = df['data_ocorrencia'].dt.hour
    # Prepara coluna dia_semana para gráficos
    df['dia_semana'] = df['data_ocorrencia'].dt.day_name().replace({
        'Monday':'Segunda','Tuesday':'Terça','Wednesday':'Quarta',
        'Thursday':'Quinta','Friday':'Sexta','Saturday':'Sábado','Sunday':'Domingo'
    })
    return df

df = carregar_dados()

# FILTROS
st.title("Mapa de Hotspots de Ocorrências")
st.markdown("Use os filtros abaixo para explorar padrões de criminalidade por bairro, tipo de crime e data.")

col_f1, col_f2 = st.columns(2)

bairros = sorted(df['bairro'].dropna().unique().tolist())
tipos_crime = sorted(df['tipo_crime'].dropna().unique().tolist())

bairro_sel = col_f1.selectbox("Selecione o Bairro", ["Todos"] + bairros)
crime_sel = col_f2.selectbox("Selecione o Tipo de Crime", ["Todos"] + tipos_crime)

st.markdown("")  # Espaçamento visual
col_data1, col_data2 = st.columns([1, 3]) # Adicionando colunas para a checkbox e o input de data

with col_data1:
    todas_datas = st.checkbox("Considerar todas as datas", value=True)

data_sel = None
if not todas_datas:
    # Garante que o valor inicial não seja um NaT se o min() falhar
    min_date = df['data_ocorrencia'].min() if not df['data_ocorrencia'].min() is pd.NaT else datetime.now().date()
    max_date = df['data_ocorrencia'].max() if not df['data_ocorrencia'].max() is pd.NaT else datetime.now().date()

    with col_data2:
        data_sel = st.date_input(
            "Selecione a Data (Desmarque 'Todas as Datas' para usar)",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )


# Aplicar filtros
df_filtrado = df.copy()
if bairro_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado['bairro'] == bairro_sel]
if crime_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado['tipo_crime'] == crime_sel]
if not todas_datas and data_sel is not None:
    # Certifica-se que a comparação é feita corretamente
    df_filtrado = df_filtrado[df_filtrado['data_ocorrencia'].dt.date == data_sel]

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados.")
    st.stop()

# --- BLOCÃO DE ANÁLISE ---
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
st.subheader("Análise Histórica e Distribuição")

# Layout para os gráficos básicos
col_g1, col_g2 = st.columns(2)

with col_g1:
    # Gráfico de Linha (Tendência Temporal) - Opção fixa para ser mais visível
    df_tempo = df_filtrado.groupby(df_filtrado['data_ocorrencia'].dt.date).size().reset_index(name='Ocorrências')
    fig_line = px.line(df_tempo, x='data_ocorrencia', y='Ocorrências', markers=True,
                       title="Tendência Temporal das Ocorrências", height=350,
                       color_discrete_sequence=['#CC3300']) # Cor para destaque
    fig_line.update_layout(xaxis_title="Data", yaxis_title="Contagem")
    st.plotly_chart(fig_line, use_container_width=True)

with col_g2:
    # NOVO GRÁFICO: Bairros Mais Perigosos (ranking por ocorrências)
    ocorrencias_bairro = df_filtrado['bairro'].value_counts().head(10).reset_index()
    ocorrencias_bairro.columns = ['Bairro', 'Ocorrências']
    
    fig_bar_bairro = px.bar(
        ocorrencias_bairro, 
        x='Ocorrências', 
        y='Bairro', 
        orientation='h',
        # Título alterado para refletir o perigo/densidade
        title="Ranking de Bairros Mais Perigosos (Por Volume de Ocorrências)", 
        height=350,
        color='Ocorrências', # Usa a contagem de ocorrências para colorir
        color_continuous_scale=px.colors.sequential.Reds # Escala de cor mais intensa
    )
    # Garante a ordem do maior para o menor (topo do gráfico é o mais perigoso)
    fig_bar_bairro.update_layout(yaxis={'categoryorder':'total ascending'}) 
    st.plotly_chart(fig_bar_bairro, use_container_width=True)


st.markdown('</div>', unsafe_allow_html=True)

# --- SEÇÃO: DETALHES TEMPORAIS E PERFIL ---
st.markdown('<div class="bloco-analise">', unsafe_allow_html=True)
st.subheader("Padrões de Ocorrência e Perfil do Suspeito")

col_g3, col_g4 = st.columns(2)

with col_g3:
    # 1. Ocorrências por Hora do Dia
    horarios = df_filtrado['hora_dia'].value_counts().sort_index().reset_index()
    horarios.columns = ['Hora do Dia', 'Ocorrências']
    fig_hora = px.bar(horarios, x='Hora do Dia', y='Ocorrências', 
                      title="Picos de Ocorrência por Hora do Dia", height=350,
                      color_discrete_sequence=['#0066CC'])
    fig_hora.update_layout(xaxis={'tickmode': 'linear', 'dtick': 2}, yaxis_title="Contagem")
    st.plotly_chart(fig_hora, use_container_width=True)

with col_g4:
    # 2. Ocorrências por Dia da Semana
    ordem_dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    dias = df_filtrado['dia_semana'].value_counts().reindex(ordem_dias, fill_value=0).reset_index()
    dias.columns = ['Dia da Semana', 'Ocorrências']
    
    fig_dia = px.bar(dias, x='Dia da Semana', y='Ocorrências', 
                     title="Distribuição por Dia da Semana", height=350,
                     color='Ocorrências', color_continuous_scale=px.colors.sequential.Sunset)
    fig_dia.update_layout(yaxis_title="Contagem")
    st.plotly_chart(fig_dia, use_container_width=True)


col_g5, col_g6 = st.columns(2)

with col_g5:
    # 3. Distribuição de Idade do Suspeito (Histrograma)
    df_idade = df_filtrado.dropna(subset=['idade_suspeito'])
    if not df_idade.empty:
        fig_idade = px.histogram(df_idade, x='idade_suspeito', nbins=20, 
                                 title="Distribuição de Idade do Suspeito", height=350,
                                 color_discrete_sequence=['#008080'])
        fig_idade.update_layout(xaxis_title="Idade", yaxis_title="Frequência")
        st.plotly_chart(fig_idade, use_container_width=True)
    else:
        st.info("Dados de idade do suspeito insuficientes para este filtro.")

with col_g6:
    # 4. Top Tipos de Arma
    armas = df_filtrado['arma_utilizada'].value_counts().head(5).reset_index()
    armas.columns = ['Arma Utilizada', 'Ocorrências']
    fig_arma = px.bar(armas, x='Ocorrências', y='Arma Utilizada', orientation='h', 
                      title="Top 5 Armas Utilizadas", height=350,
                      color='Arma Utilizada', color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_arma.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_arma, use_container_width=True)


st.markdown('</div>', unsafe_allow_html=True)


# MAPA DE CALOR (no final)
st.subheader("Mapa de Calor de Ocorrências")

if 'latitude' not in df_filtrado.columns or 'longitude' not in df_filtrado.columns:
    st.error("O dataset precisa conter as colunas 'latitude' e 'longitude' para gerar o mapa.")
else:
    # Centraliza o mapa na média dos dados filtrados
    map_center = [df_filtrado['latitude'].mean(), df_filtrado['longitude'].mean()]

    mapa = folium.Map(
        location=map_center,
        zoom_start=12, # Zoom um pouco mais aberto para visualização inicial
        tiles='CartoDB positron'
    )
    
    # Prepara os dados para o HeatMap, garantindo que sejam válidos
    heat_data = df_filtrado[['latitude', 'longitude']].dropna().values.tolist()
    
    if heat_data:
        HeatMap(heat_data, radius=15, blur=20, min_opacity=0.4).add_to(mapa)
        st_folium(mapa, width=1150, height=550)
    else:
        st.warning("Não há dados de latitude e longitude válidos para o mapa com os filtros aplicados.")