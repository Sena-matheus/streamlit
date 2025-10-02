import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import re

st.set_page_config(page_title="Delegacia Inteligente", layout="wide")

st.title("Delegacia Inteligente")
st.write(
    "Upload e pré-visualização de dataset. Esta base será utilizada nas próximas páginas: Hotspots, Clusters, Anomalias e Modelo Supervisionado."
)

# Sidebar: upload e opções
st.sidebar.header("Upload")
sample_choice = st.sidebar.selectbox("Escolher dataset de exemplo", ("—", "Exemplo sintético"))
uploaded_file = st.sidebar.file_uploader("Upload (CSV / XLSX)", type=["csv", "xlsx"])

@st.cache_data
def load_data_from_file(uploaded_file):
    if uploaded_file is None:
        return None
    name = getattr(uploaded_file, "name", "").lower()
    try:
        if name.endswith('.csv') or uploaded_file.type == 'text/csv':
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None

@st.cache_data
def create_example_df():
    n = 300
    df = pd.DataFrame({
        "latitude": -8.05 + np.random.randn(n) * 0.015,
        "longitude": -34.9 + np.random.randn(n) * 0.015,
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="H"),
        "categoria": np.random.choice(["Roubo", "Furto", "Assalto"], size=n)
    })
    return df

# carregar dataframe
if uploaded_file is not None:
    df = load_data_from_file(uploaded_file)
elif sample_choice == "Exemplo sintético":
    df = create_example_df()
else:
    df = None

if df is None:
    st.info("Carregue um CSV/XLSX pela barra lateral ou escolha o exemplo sintético para começar.")
else:
    # Seleção de coluna pelo usuário
    selected_col = st.selectbox("Selecionar uma Coluna", options=["(nenhuma)"] + list(df.columns))

    if selected_col != "(nenhuma)":
        st.subheader(f"Resumo da coluna: {selected_col}")
        col_data = df[selected_col]

        if pd.api.types.is_numeric_dtype(col_data):
            st.write("Tipo: Numérico")
            st.write(f"Média: {col_data.mean():.2f}")
            st.write(f"Mediana: {col_data.median():.2f}")
            st.write(f"Mínimo: {col_data.min():.2f}")
            st.write(f"Máximo: {col_data.max():.2f}")
            st.write("Alguns valores aleatórios:")
            st.write(col_data.sample(min(5, len(col_data))).values)

        elif pd.api.types.is_datetime64_any_dtype(col_data):
            st.write("Tipo: Data/Hora")
            st.write(f"Primeiro registro: {col_data.min()}")
            st.write(f"Último registro: {col_data.max()}")
            st.write(f"Número de registros: {len(col_data)}")
            st.write("Alguns exemplos aleatórios:")
            st.write(col_data.sample(min(5, len(col_data))).dt.strftime("%Y-%m-%d %H:%M:%S").values)

        else:
            st.write("Tipo: Categórico / Texto")
            st.write("Valores mais frequentes:")
            st.write(col_data.value_counts().head())
            st.write("Alguns exemplos aleatórios:")
            st.write(col_data.sample(min(5, len(col_data))).values)

    # Pré-visualização com nomes “limpos”
    def clean_column_names(df):
        df_clean = df.copy()
        df_clean.columns = [re.sub(r'\W+', '_', c) for c in df_clean.columns]  # substitui caracteres não alfanuméricos por '_'
        return df_clean

    with st.expander("Pré-visualização das primeiras linhas"):
        st.dataframe(clean_column_names(df).head(), use_container_width=True)

    # Estatísticas gerais
    with st.expander("Estatísticas gerais"):
        try:
            st.write(df.describe(include='all'))
        except Exception:
            st.write("Não foi possível gerar estatísticas para este dataset.")

    # Resumo de valores nulos
    st.subheader("Resumo de valores nulos por coluna")
    st.write(df.isnull().sum())

    # Filtros interativos para colunas categóricas
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    for col in categorical_cols:
        if st.sidebar.checkbox(f"Filtrar {col}?"):
            options = st.sidebar.multiselect(f"Selecionar valores de {col}", df[col].unique(), default=df[col].unique())
            df = df[df[col].isin(options)]

    # Gráficos simples de distribuição
    if categorical_cols:
        st.subheader("Distribuição das categorias")
        for col in categorical_cols:
            counts = df[col].value_counts()
            fig, ax = plt.subplots()
            ax.bar(counts.index, counts.values)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # Salvar dataframe em session_state para próximas páginas
    st.session_state['df'] = df

    # Botão para baixar CSV pré-processado
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar CSV (pré-processado)", data=csv, file_name="dataset_preparado.csv", mime="text/csv")

    st.markdown("---")
    st.info("Próximos passos: criar páginas para Hotspots, Clusters, Anomalias e Modelo Supervisionado.")
