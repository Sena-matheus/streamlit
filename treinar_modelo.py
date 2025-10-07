import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1️⃣ Carregar o dataset
df = pd.read_csv("dataset_ocorrencias_delegacia_5.csv")

# 2️⃣ Tratar datas e criar colunas derivadas
df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'])

# Criar dia da semana em português
df['dia_semana'] = df['data_ocorrencia'].dt.day_name().replace({
    'Monday': 'Segunda',
    'Tuesday': 'Terça',
    'Wednesday': 'Quarta',
    'Thursday': 'Quinta',
    'Friday': 'Sexta',
    'Saturday': 'Sábado',
    'Sunday': 'Domingo'
})

# Criar hora do dia
df['hora_dia'] = df['data_ocorrencia'].dt.hour

# 3️⃣ Selecionar features e target
X = df[[
    "bairro",
    "arma_utilizada",
    "quantidade_vitimas",
    "quantidade_suspeitos",
    "sexo_suspeito",
    "idade_suspeito",
    "dia_semana",
    "hora_dia",
    "latitude",
    "longitude"
]]

y = df["tipo_crime"]

# 4️⃣ Tratar valores nulos
X = X.fillna("Desconhecido")
y = y.fillna("Desconhecido")

# 5️⃣ One-hot encoding para variáveis categóricas
X = pd.get_dummies(X, drop_first=True)

# 6️⃣ Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 7️⃣ Treinar modelo Random Forest
modelo = RandomForestClassifier(n_estimators=200, random_state=42)
modelo.fit(X_train, y_train)

# 8️⃣ Avaliar modelo
y_pred = modelo.predict(X_test)
print("✅ Relatório de classificação do modelo:")
print(classification_report(y_test, y_pred))

# 9️⃣ Salvar modelo e colunas
joblib.dump(modelo, "modelo.pkl")
joblib.dump(X.columns, "colunas.pkl")

print("✅ Modelo e colunas salvos com sucesso!")
