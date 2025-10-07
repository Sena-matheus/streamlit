import streamlit as st
import pandas as pd
from datetime import datetime
import uuid

# Configuração da página 
st.set_page_config(
    page_title="Delegacia Inteligente - Home",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# Função para carregar os dados
@st.cache_data
def carregar_dados():
    # Adicionando um fallback simples caso o arquivo não esteja na raiz
    try:
        df = pd.read_csv("dataset_ocorrencias_delegacia_5.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("../dataset_ocorrencias_delegacia_5.csv")
        except FileNotFoundError:
            st.error("Arquivo 'dataset_ocorrencias_delegacia_5.csv' não encontrado.")
            st.stop()
    return df

# Inicializa o DataFrame e converte a coluna de data
df = carregar_dados()
df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'], errors='coerce')


# Métricas principais
st.header("Resumo Geral dos Dados")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de Registros", f"{len(df):,}".replace(",", "."))
with col2:
    data_min = df['data_ocorrencia'].min()
    data_max = df['data_ocorrencia'].max()
    periodo_dias = (data_max - data_min).days if not (pd.isnull(data_min) or pd.isnull(data_max)) else 0
    st.metric("Período Coberto (Dias)", periodo_dias)
with col3:
    st.metric("Total de Bairros Únicos", df['bairro'].nunique())
with col4:
    st.metric("Total de Tipos de Crime", df['tipo_crime'].nunique())

st.divider()

# Informes Hierárquicos e Interativos
st.subheader("Informes Rápidos e Análise Hierárquica")

st.markdown(
    "**Estes informes contam uma história sobre os crimes por bairro, crime e data.** "
    "Escolha um bairro e, opcionalmente, filtre por crime e data para obter detalhes."
)

# Opções de filtro
bairros = sorted(df['bairro'].dropna().unique().tolist())
bairro_selecionado = st.selectbox("1. Escolha o Bairro", bairros)

# Filtro 1: Apenas o Bairro
df_bairro = df[df['bairro'] == bairro_selecionado].copy()

# Filtro de data
todas_datas = st.checkbox("Considerar todas as datas", value=True, key='todas_datas_home')
data_selecionada = None

if not todas_datas:
    # Garante que as datas sejam manipuladas corretamente
    data_min_val = df_bairro['data_ocorrencia'].min().date() if not df_bairro['data_ocorrencia'].min() is pd.NaT else datetime.now().date()
    data_max_val = df_bairro['data_ocorrencia'].max().date() if not df_bairro['data_ocorrencia'].max() is pd.NaT else datetime.now().date()
    
    data_selecionada = st.date_input(
        "2. Selecione a Data da Ocorrência",
        value=data_min_val,
        min_value=data_min_val,
        max_value=data_max_val
    )
    
    # Aplica filtro de data no DataFrame
    df_bairro = df_bairro[df_bairro['data_ocorrencia'].dt.date == data_selecionada]

if not df_bairro.empty:
    # Cartões de Resumo do Bairro/Data
    col1, col2, col3 = st.columns(3)

    crime_mais_frequente = df_bairro['tipo_crime'].mode().iloc[0] if not df_bairro['tipo_crime'].mode().empty else 'N/A'
    col1.info(f"**Crime Mais Frequente**: {crime_mais_frequente}")

    # Garante que o cálculo da hora seja feito em dados válidos
    horario_mais_frequente = df_bairro['data_ocorrencia'].dt.hour.mode().iloc[0] if not df_bairro['data_ocorrencia'].dt.hour.mode().empty else 'N/A'
    col2.info(f"**Horário Mais Frequente**: {horario_mais_frequente}:00")

    total_registros_filtrados = len(df_bairro)
    col3.success(f"**Total de Ocorrências**: {total_registros_filtrados}")

    # Filtro 3: Crime Específico
    top_crimes = df_bairro['tipo_crime'].value_counts().head(5).index.tolist()
    crime_selecionado = st.selectbox(
        "3. Filtrar por Crime (opcional)",
        ["Todos"] + top_crimes
    )

    if crime_selecionado != "Todos":
        df_crime = df_bairro[df_bairro['tipo_crime'] == crime_selecionado].copy()
        titulo_detalhes = f"### Detalhes para {bairro_selecionado} - {crime_selecionado}" + (f" em {data_selecionada}" if not todas_datas else "")
        st.markdown(titulo_detalhes)

        # Resumo do Crime Específico
        col_c1, col_c2, col_c3 = st.columns(3)
        
        if 'arma_utilizada' in df_crime.columns:
             arma_mais_comum = df_crime['arma_utilizada'].mode().iloc[0] if not df_crime['arma_utilizada'].mode().empty else 'N/A'
             col_c1.info(f"**Arma Mais Comum**: {arma_mais_comum}")

        if 'sexo_suspeito' in df_crime.columns:
            sexo_mais_frequente = df_crime['sexo_suspeito'].mode().iloc[0] if not df_crime['sexo_suspeito'].mode().empty else 'N/A'
            col_c2.info(f"**Sexo do Suspeito**: {sexo_mais_frequente}")

        horario_medio = df_crime['data_ocorrencia'].dt.hour.mean()
        col_c3.info(f"**Horário Médio do Crime**: {horario_medio:.2f}h")

        # Mostrar tabela de ocorrências detalhadas
        colunas_exibir = ['data_ocorrencia', 'bairro', 'tipo_crime']
        if 'arma_utilizada' in df_crime.columns:
            colunas_exibir.append('arma_utilizada')
        if 'sexo_suspeito' in df_crime.columns:
            colunas_exibir.append('sexo_suspeito')

        st.markdown("##### Ocorrências Detalhadas (Amostra):")
        st.dataframe(df_crime[colunas_exibir].head(10), use_container_width=True)
        
    else:
        # Detalhes do Bairro/Data sem filtro de Crime
        titulo_detalhes = f"### Resumo dos Principais Crimes em {bairro_selecionado}" + (f" em {data_selecionada}" if not todas_datas else "")
        st.markdown(titulo_detalhes)
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.markdown("##### Top 5 Tipos de Crime:")
            top_crimes_tabela = df_bairro['tipo_crime'].value_counts().head(5)
            st.table(top_crimes_tabela)

        with col_r2:
            st.markdown("##### Horário Médio por Crime (Top 5):")
            horario_medio_por_crime = df_bairro.groupby('tipo_crime')['data_ocorrencia'].apply(lambda x: x.dt.hour.mean()).reset_index()
            horario_medio_por_crime.columns = ['Tipo de Crime', 'Horário Médio']
            # Filtra apenas para os Top 5 Crimes para manter o foco
            horario_medio_por_crime = horario_medio_por_crime[horario_medio_por_crime['Tipo de Crime'].isin(top_crimes)].set_index('Tipo de Crime')
            st.table(horario_medio_por_crime.style.format({'Horário Médio': '{:.2f}h'}))

else:
    st.warning("Não há ocorrências registradas para esta combinação de bairro/data.")
    
# Cadastro de Ocorrências
st.divider()
st.header("Cadastro de Nova Ocorrência Criminal")
st.markdown("Use o formulário abaixo para simular o registro de um novo Boletim de Ocorrência (BO).")

# Extrai listas únicas de colunas categóricas para popular os selects
bairros_unicos = sorted(df['bairro'].dropna().unique().tolist())
tipos_crime_unicos = sorted(df['tipo_crime'].dropna().unique().tolist())
opcoes_arma = sorted(df['arma_utilizada'].dropna().unique().tolist()) if 'arma_utilizada' in df.columns else ['N/A', 'Faca', 'Revólver', 'Outro']
opcoes_sexo = sorted(df['sexo_suspeito'].dropna().unique().tolist()) if 'sexo_suspeito' in df.columns else ['Não Informado', 'MASCULINO', 'FEMININO']

with st.form("cadastro_ocorrencia"):
    st.subheader("Dados da Ocorrência")
    
    # Coluna 1: Data e Hora
    col_d_t, col_b = st.columns(2)
    
    data_reg = col_d_t.date_input("Data da Ocorrência", datetime.now().date())
    hora_reg = col_d_t.time_input("Hora da Ocorrência", datetime.now().time())
    
    # Coluna 2: Local e Tipo de Crime
    bairro_reg = col_b.selectbox("Bairro da Ocorrência", bairros_unicos)
    tipo_crime_reg = col_b.selectbox("Tipo de Crime", tipos_crime_unicos)
    
    # Coluna 3: Modus Operandi
    col_mo1, col_mo2 = st.columns(2)
    
    arma_reg = col_mo1.selectbox("Arma Utilizada", opcoes_arma)
    sexo_reg = col_mo2.selectbox("Sexo do Suspeito", opcoes_sexo)
    
    observacoes = st.text_area("Observações Adicionais", height=100)
    
    submitted = st.form_submit_button("Registrar Ocorrência")
    
    if submitted:
        # Combina data e hora
        data_hora_ocorrencia = datetime.combine(data_reg, hora_reg)
        
        # Simula o registro
        novo_registro = {
            'id_ocorrencia': str(uuid.uuid4()),
            'data_ocorrencia': data_hora_ocorrencia.strftime('%Y-%m-%d %H:%M:%S'),
            'bairro': bairro_reg,
            'tipo_crime': tipo_crime_reg,
            'arma_utilizada': arma_reg,
            'sexo_suspeito': sexo_reg,
            'observacoes': observacoes
        }
        
        st.success(f"Ocorrência registrada com sucesso!")
        st.json(novo_registro)
        st.info("Para fins de demonstração, o registro foi apenas exibido e **não salvo permanentemente** no arquivo CSV.")