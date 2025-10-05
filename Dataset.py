import streamlit as st
import pandas as pd

# Configuração da página
st.set_page_config(
    page_title="Delegacia Inteligente - Home",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# Título principal
st.title("Página Inicial - DELIT")
st.markdown(
    "Bem-vindo ao painel de ocorrências do DELIT. Aqui você pode visualizar um resumo geral dos dados disponíveis no sistema."
)

# Carregar dataset fixo
@st.cache_data
def carregar_dados():
    return pd.read_csv("dataset_ocorrencias_delegacia_5.csv")

df = carregar_dados()

# Métricas principais
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de Registros", f"{len(df):,}".replace(",", "."))
with col2:
    st.metric("Total de Colunas", len(df.columns))
with col3:
    st.metric("Colunas Numéricas", df.select_dtypes(include=['number']).shape[1])
with col4:
    st.metric("Colunas Categóricas", df.select_dtypes(exclude=['number']).shape[1])

st.divider()

# Pré-visualização do Dataset
st.subheader("Pré-visualização dos Dados")
st.dataframe(df.head(), use_container_width=True)

# Estatísticas Gerais
st.subheader("Estatísticas Gerais")
st.dataframe(df.describe().T, use_container_width=True)

st.divider()

# Informes Hierárquicos e Interativos
st.divider()
st.subheader("Informes Rápidos")

st.markdown(
    "**Estes informes contam uma história sobre os crimes por bairro, crime e data.** "
    "Escolha um bairro e, opcionalmente, um crime e/ou uma data para explorar os dados de forma hierárquica."
)

def nome_formal(coluna):
    return coluna.replace("_", " ").title()

df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'])

bairros = df['bairro'].unique().tolist()
bairro_selecionado = st.selectbox("Escolha o Bairro", bairros)

df_bairro = df[df['bairro'] == bairro_selecionado]

st.markdown(
    """
    <style>
    div[data-baseweb="datepicker"] > div > input {
        border: 2px solid #00acc1 !important;
        border-radius: 8px !important;
        padding: 8px !important;
        box-shadow: 0 0 10px rgba(0, 172, 193, 0.5) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

todas_datas = st.checkbox("Considerar todas as datas", value=True)
if not todas_datas:
    data_selecionada = st.date_input(
        "Selecione a Data da Ocorrência",
        value=df_bairro['data_ocorrencia'].min(),
        min_value=df_bairro['data_ocorrencia'].min(),
        max_value=df_bairro['data_ocorrencia'].max()
    )
    df_bairro = df_bairro[df_bairro['data_ocorrencia'].dt.date == data_selecionada]

if not df_bairro.empty:
    col1, col2, col3 = st.columns(3)

    crime_mais_frequente = df_bairro['tipo_crime'].mode()[0]
    col1.info(f"**Crime Mais Frequente**: {crime_mais_frequente}")

    horario_mais_frequente = df_bairro['data_ocorrencia'].dt.hour.mode()[0]
    col2.info(f"**Horário Mais Frequente**: {horario_mais_frequente}:00")

    colunas_numericas = df_bairro.select_dtypes(include=['number']).columns
    if len(colunas_numericas) > 0:
        media_bairro = df_bairro[colunas_numericas].mean().mean()
        col3.markdown(
            f"<div style='background-color:#FFD700; padding:10px; border-radius:5px;'>"
            f"Média Geral dos Valores Numéricos: {media_bairro:.2f}</div>",
            unsafe_allow_html=True
        )

    top_crimes = df_bairro['tipo_crime'].value_counts().head(5).index.tolist()
    crime_selecionado = st.selectbox(
        "Filtrar por Crime (opcional)",
        ["Todos"] + top_crimes
    )

    if crime_selecionado != "Todos":
        df_crime = df_bairro[df_bairro['tipo_crime'] == crime_selecionado]
        st.markdown(f"### Detalhes para {bairro_selecionado} - {crime_selecionado}" + (f" em {data_selecionada}" if not todas_datas else ""))

        # Tipo de arma mais comum
        if 'tipo_arma' in df_crime.columns:
            arma_mais_comum = df_crime['tipo_arma'].mode()[0]
            st.info(f"**Tipo de Arma Mais Comum**: {arma_mais_comum}")

        # Sexo do suspeito mais frequente
        if 'sexo_suspeito' in df_crime.columns:
            sexo_mais_frequente = df_crime['sexo_suspeito'].mode()[0]
            st.info(f"**Sexo do Suspeito Mais Frequente**: {sexo_mais_frequente}")

        # Horário médio do crime
        horario_medio = df_crime['data_ocorrencia'].dt.hour.mean()
        st.info(f"**Horário Médio do Crime**: {horario_medio:.2f}h")

        # Mostrar tabela de ocorrências detalhadas
        colunas_exibir = ['data_ocorrencia', 'bairro', 'tipo_crime']
        if 'tipo_arma' in df_crime.columns:
            colunas_exibir.append('tipo_arma')
        if 'sexo_suspeito' in df_crime.columns:
            colunas_exibir.append('sexo_suspeito')

        st.dataframe(df_crime[colunas_exibir])
    else:
        st.markdown(f"### Detalhes para {bairro_selecionado}" + (f" em {data_selecionada}" if not todas_datas else ""))
        top_crimes_tabela = df_bairro['tipo_crime'].value_counts().head(5)
        st.table(top_crimes_tabela)

        horario_medio_por_crime = df_bairro.groupby('tipo_crime')['data_ocorrencia'].apply(lambda x: x.dt.hour.mean()).reset_index()
        horario_medio_por_crime.columns = ['Tipo de Crime', 'Horário Médio']
        st.table(horario_medio_por_crime)

else:
    st.warning("Não há ocorrências registradas para esta combinação de bairro/crime/data.")


