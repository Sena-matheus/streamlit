import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pydeck as pdk 

sns.set_style("whitegrid")

st.set_page_config(page_title="Predição Estratégica de Crimes", layout="wide")
st.title("Predição Estratégica de Crimes - DELIT")
st.markdown("Ferramenta de previsão para planejamento policial baseado em Machine Learning.")

# Carregar dataset e modelo
@st.cache_data
def carregar_dados():
    # OBSERVAÇÃO: Mantenha seu carregamento de dados original aqui
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

# Carregar modelo e colunas
try:
    modelo = joblib.load("modelo.pkl")
    colunas = joblib.load("colunas.pkl")
    classes_modelo = modelo.classes_
except FileNotFoundError:
    st.error("Erro: Arquivos 'modelo.pkl' ou 'colunas.pkl' não encontrados. Verifique os caminhos.")
    st.stop()
except AttributeError:
    # Garantia em caso de erro no modelo, para que a execução continue
    classes_modelo = np.array(['Crime A', 'Crime B', 'Crime C', 'Crime D']) 

# --- Seleção do horizonte de previsão ---
st.subheader("Escolha o horizonte de previsão")
horizonte = st.selectbox(
    "Horizonte de previsão",
    ("Amanhã", "Próxima Semana", "Próximo Mês", "Próximo Semestre")
)

hoje = datetime.now().date() # Uso .date() para facilitar a criação do range de datas
# Lógica do horizonte (ajustada para datetime.date)
if horizonte == "Amanhã":
    inicio_data = hoje + timedelta(days=1)
    fim_data = inicio_data
elif horizonte == "Próxima Semana":
    inicio_data = hoje + timedelta(days=1)
    fim_data = hoje + timedelta(days=7)
elif horizonte == "Próximo Mês":
    # Adicionando 1 mês ao invés de usar pd.DateOffset que retorna datetime
    inicio_data = (datetime.now() + pd.DateOffset(months=1)).date()
    fim_data = (datetime.now() + pd.DateOffset(months=2) - timedelta(days=1)).date()
elif horizonte == "Próximo Semestre":
    inicio_data = (datetime.now() + pd.DateOffset(months=1)).date()
    fim_data = (datetime.now() + pd.DateOffset(months=6) - timedelta(days=1)).date()

# Reconvertendo para datetime para o range
inicio = datetime.combine(inicio_data, datetime.min.time())
fim = datetime.combine(fim_data, datetime.min.time())


# --- Filtros interativos no corpo da página ---
st.subheader("Filtros Interativos")

# Primeira linha de filtros (Local e Tempo)
col1, col2, col3 = st.columns(3)
with col1:
    bairros = df['bairro'].unique()
    bairro_selecionado = st.multiselect("Bairro", bairros, default=[]) 
with col2:
    dias_semana = df['dia_semana'].unique()
    dia_selecionado = st.multiselect("Dia da Semana", dias_semana, default=[]) 
with col3:
    horas = range(0,24)
    hora_selecionada = st.slider("Hora do Dia", 0, 23, (0,23))

# Segunda linha de filtros (Características do Crime)
col4, col5, col6, col7 = st.columns(4)
with col4:
    crimes = df['tipo_crime'].unique()
    crime_selecionado = st.multiselect("Tipo de Crime Histórico", crimes, default=[]) 
with col5:
    armas = df['arma_utilizada'].unique()
    arma_selecionada = st.multiselect("Arma Utilizada", armas, default=[])
with col6:
    generos = df['sexo_suspeito'].dropna().unique()
    genero_selecionado = st.multiselect("Gênero Suspeito", generos, default=[])
with col7:
    idade_min = int(df['idade_suspeito'].min()) if not df['idade_suspeito'].empty and not pd.isna(df['idade_suspeito'].min()) else 18
    idade_max = int(df['idade_suspeito'].max()) if not df['idade_suspeito'].empty and not pd.isna(df['idade_suspeito'].max()) else 60
    idade_selecionada = st.slider("Idade do Suspeito", idade_min, idade_max, (idade_min, idade_max))


# --- Filtrando o dataset histórico relevante ---
df_filtrado = df[
    (df['bairro'].isin(bairro_selecionado) if bairro_selecionado else True) &
    (df['dia_semana'].isin(dia_selecionado) if dia_selecionado else True) &
    (df['tipo_crime'].isin(crime_selecionado) if crime_selecionado else True) &
    (df['arma_utilizada'].isin(arma_selecionada) if arma_selecionada else True) &
    (df['sexo_suspeito'].isin(genero_selecionado) if genero_selecionado else True) &
    (df['idade_suspeito'].between(idade_selecionada[0], idade_selecionada[1])) &
    (df['hora_dia'].between(hora_selecionada[0], hora_selecionada[1]))
]

# --- Previsão de Top N Crimes Mais Prováveis ---
st.subheader(f"Previsão Estratégica para: {horizonte}")

if df_filtrado.empty:
    st.warning("Não há dados históricos suficientes para gerar previsão com os filtros selecionados.")
else:
    # Tratamento de dados: usamos dropna() para garantir que a média e moda não falhem
    df_limpo = df_filtrado.dropna(subset=['quantidade_vitimas', 'quantidade_suspeitos', 'idade_suspeito'])
    
    if df_limpo.empty:
        st.warning("Os filtros selecionados resultam em dados vazios após a limpeza de valores ausentes.")
    else:
        # Criação da entrada de dados com as médias/modas do histórico filtrado (base para a previsão)
        entrada_base = {
            'bairro': df_filtrado['bairro'].mode()[0],
            'arma_utilizada': df_filtrado['arma_utilizada'].mode()[0],
            'quantidade_vitimas': int(df_limpo['quantidade_vitimas'].mean()),
            'quantidade_suspeitos': int(df_limpo['quantidade_suspeitos'].mean()),
            'sexo_suspeito': df_filtrado['sexo_suspeito'].mode()[0] if not df_filtrado['sexo_suspeito'].mode().empty else 'NA',
            'idade_suspeito': int(df_limpo['idade_suspeito'].mean()),
            'latitude': df_filtrado['latitude'].mean(),
            'longitude': df_filtrado['longitude'].mean()
        }
        
        # --- Previsão de Top 3 (Para o texto principal) ---
        # A previsão do Top 3 deve ser feita usando as horas e dias mais prováveis/filtrados
        entrada_media_top3 = entrada_base.copy()
        entrada_media_top3['dia_semana'] = df_filtrado['dia_semana'].mode()[0]
        entrada_media_top3['hora_dia'] = int(df_filtrado['hora_dia'].mean())
        
        entrada_df_top3 = pd.DataFrame([entrada_media_top3])
        entrada_df_top3 = pd.get_dummies(entrada_df_top3)
        entrada_df_top3 = entrada_df_top3.reindex(columns=colunas, fill_value=0)

        # Usando predict_proba para obter as probabilidades
        probabilidades = modelo.predict_proba(entrada_df_top3)[0]
        
        # Cria um DataFrame de resultados
        df_prob = pd.DataFrame({
            'Tipo de Crime': classes_modelo,
            'Probabilidade': probabilidades
        }).sort_values('Probabilidade', ascending=False).head(3)
        
        # Apresentação do Top 3
        st.markdown(f"**Tipo de crime mais provável:** **{df_prob.iloc[0]['Tipo de Crime']}**")
        st.markdown("---")
        st.info("Top 3 Tipos de Crime Mais Prováveis (Baseado nas Características Filtradas):")
        
        # Formata a probabilidade para percentual
        df_prob['Probabilidade'] = (df_prob['Probabilidade'] * 100).round(1).astype(str) + '%'
        
        st.table(df_prob.reset_index(drop=True))

        # --- Geração de Dados Sintéticos de Previsão para os Gráficos ---
        
        # Mapeamento de dias da semana para uso no loop (NOVA LINHA)
        dias_pt = {
            'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta',
            'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
        }

        # 1. Cria um range de datas e horas para o horizonte (amostragem)
        # Para evitar processamento muito longo, criamos uma amostra razoável
        if horizonte in ["Próximo Mês", "Próximo Semestre"]:
            # Amostra a cada 6 horas para meses/semestres
            freq = '6H' 
        else:
            # Amostra a cada 1 hora para dias/semanas
            freq = 'H'
            
        data_range = pd.date_range(start=inicio, end=fim, freq=freq)
        
        # Se o range for muito pequeno (e.g., Amanhã), vamos replicar para simular mais "ocorrências"
        if len(data_range) < 24:
             data_range = pd.date_range(start=inicio, end=fim + timedelta(hours=23), freq='H')

        
        previsao_list = []
        
        # 2. Loop para fazer a previsão para cada ponto no tempo
        for dt in data_range:
            # Pega o nome do dia da semana (que é uma string)
            dia_semana_en = dt.day_name()
            # Mapeia a string para o português usando .get() - CORREÇÃO
            dia_semana_prev = dias_pt.get(dia_semana_en, dia_semana_en)
            
            hora_dia_prev = dt.hour
            
            # Cria a linha de entrada com as características do filtro + o tempo atual
            entrada_prev = entrada_base.copy()
            entrada_prev['dia_semana'] = dia_semana_prev
            entrada_prev['hora_dia'] = hora_dia_prev
            
            # Converte para DataFrame e faz One-Hot Encoding
            df_entrada_prev = pd.DataFrame([entrada_prev])
            df_entrada_prev = pd.get_dummies(df_entrada_prev)
            df_entrada_prev = df_entrada_prev.reindex(columns=colunas, fill_value=0)
            
            # 3. Faz a previsão
            pred_crime = modelo.predict(df_entrada_prev)[0]
            
            previsao_list.append({
                'tipo_crime_previsto': pred_crime,
                'dia_semana_previsto': dia_semana_prev,
                'hora_dia_previsto': hora_dia_prev
            })
            
        # Cria o DataFrame de previsões
        df_previsao = pd.DataFrame(previsao_list)


# --- Gráficos de Previsão Interativos ---
st.subheader(f"Predição de Ocorrências para o {horizonte}")

# Verifica se o df_previsao foi criado (ou se df_filtrado estava vazio)
if 'df_previsao' not in locals():
    # Isso só acontecerá se df_filtrado estiver vazio ou df_limpo estiver vazio
    pass 
else:
    col_g1, col_g2 = st.columns(2)
    col_g3, col_g4 = st.columns(2)

    # Top crimes previstos
    with col_g1:
        top_crimes_prev = df_previsao['tipo_crime_previsto'].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(8,3))
        sns.barplot(x=top_crimes_prev.values, y=top_crimes_prev.index, palette="Reds_r", ax=ax)
        ax.set_xlabel("Frequência de Previsão")
        ax.set_ylabel("")
        ax.set_title("Top 5 Crimes Previstos")
        st.pyplot(fig)
        plt.close(fig)

    # Previsões por dia da semana (garante a ordem correta)
    with col_g2:
        ordem_dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        dias_prev = df_previsao['dia_semana_previsto'].value_counts().reindex(ordem_dias, fill_value=0)
        
        # Remove dias sem previsão do gráfico
        dias_prev = dias_prev[dias_prev > 0] 
        
        fig, ax = plt.subplots(figsize=(8,3))
        sns.barplot(x=dias_prev.index, y=dias_prev.values, palette="Greens_r", ax=ax)
        ax.set_xlabel("Dia da Semana")
        ax.set_ylabel("Frequência de Previsão")
        ax.set_title("Previsões por Dia da Semana")
        st.pyplot(fig)
        plt.close(fig)

    # Frequência de Tipos de Crime por Arma (Baseado no Histórico Filtrado, mas exibido aqui)
    with col_g3:
        # Nota: O modelo prevê o crime, mas a arma é uma das variáveis de entrada. 
        # Manter este gráfico histórico filtrado é útil para entender o *contexto* da previsão.
        armas = df_filtrado['arma_utilizada'].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(8,3))
        sns.barplot(x=armas.values, y=armas.index, palette="Blues_r", ax=ax)
        ax.set_xlabel("Ocorrências Históricas")
        ax.set_ylabel("")
        ax.set_title("Top 5 Armas Utilizadas (Contexto)")
        st.pyplot(fig)
        plt.close(fig)

    # Previsões por hora do dia
    with col_g4:
        horarios_prev = df_previsao['hora_dia_previsto'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8,3))
        sns.lineplot(x=horarios_prev.index, y=horarios_prev.values, color="purple", ax=ax)
        ax.fill_between(horarios_prev.index, horarios_prev.values, alpha=0.3, color="purple")
        ax.set_xlabel("Hora do Dia")
        ax.set_ylabel("Frequência de Previsão")
        ax.set_title("Pico de Ocorrências Previstas por Hora")
        st.pyplot(fig)
        plt.close(fig)


# --- Mapas Lado a Lado ---
st.subheader("Visualização Geográfica (Ocorrências Filtradas)")

# Garantindo que o dataframe para o mapa só tenha lat/lon válidas
map_data = df_filtrado[['latitude', 'longitude']].dropna()

if map_data.empty:
    st.warning("Não há dados de latitude/longitude válidos para exibir os mapas com os filtros selecionados.")
else:
    # Define o estado de visualização centralizado na média dos dados filtrados
    view_state = pdk.ViewState(
        latitude=map_data['latitude'].mean(),
        longitude=map_data['longitude'].mean(),
        zoom=11,
        pitch=0
    )

    # Cria a Layer de Mapa de Calor
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=map_data,
        opacity=0.8,
        get_position=["longitude", "latitude"],
        threshold=0.1, 
        aggregation=pdk.types.String("SUM"),
        radius_pixels=50
    )

    # Cria as duas colunas para os mapas
    col_mapa_calor, col_mapa_pontos = st.columns(2)

    with col_mapa_calor:
        st.markdown("**Mapa de Calor (Densidade de Ocorrências Históricas)**")
        # Renderiza o mapa de calor (pydeck)
        r = pdk.Deck(
            layers=[heatmap_layer],
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/light-v9"
        )
        st.pydeck_chart(r)

    with col_mapa_pontos:
        st.markdown("**Mapa de Ocorrências (Localização Exata Histórica)**")
        # Renderiza o mapa de pontos (st.map)
        st.map(map_data)