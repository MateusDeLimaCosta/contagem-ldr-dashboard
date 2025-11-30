# ============================================================
# DASHBOARD DE CONTAGEM DE PASSAGENS – SENSOR LDR
# ============================================================
# O dashboard funciona como apresentação ("slide")
# Inclui:
#  - KPIs gerais
#  - Série temporal
#  - Heatmap
#  - Interarrival
#  - Previsão Baseline (weekday x hour)
#  - Interpretações automáticas (até a parte 5)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import csv
import io

st.set_page_config(page_title="Dashboard - Contagem LDR", layout="wide")
st.title("📡 Dashboard de Contagem de Passagens — Sensor LDR")

st.markdown("""
Este dashboard apresenta e interpreta os dados coletados pelo sensor LDR instalado na portaria.
O laser incide no LDR e, quando é interrompido, conta-se uma passagem.
Ele será usado para **apresentação do projeto**.
""")

# ============================================================
# SIDEBAR – UPLOAD DE ARQUIVO
# ============================================================

st.sidebar.header("Carregar CSV")
uploaded = st.sidebar.file_uploader("Arquivo CSV", type=["csv"])

def load_csv(file):
    raw = file.read().decode("latin-1")
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(raw.split("\n")[0], delimiters=";,")
    df = pd.read_csv(io.StringIO(raw), sep=dialect.delimiter, quotechar='"', engine="python")
    return df

if uploaded is None:
    st.warning("Envie um arquivo CSV para iniciar a análise.")
    st.stop()

df = load_csv(uploaded)

# Normalização
df.columns = df.columns.str.strip().str.lower().str.replace('"', "")

expected = ["id", "passagens", "timestamp"]
if not set(expected).issubset(df.columns):
    st.error(f"CSV inválido. Colunas encontradas: {list(df.columns)}. Esperado: {expected}")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.floor("min")
df["weekday"] = df["timestamp"].dt.day_name()
df["weekday_num"] = df["timestamp"].dt.weekday

df["prev_ts"] = df["timestamp"].shift(1)
df["prev_id"] = df["id"].shift(1)
df["delta_s"] = (df["timestamp"] - df["prev_ts"]).dt.total_seconds()
df.loc[df["id"] != df["prev_id"], "delta_s"] = np.nan

# ============================================================
# KPIs
# ============================================================

st.header("📊 Visão Geral dos Dados")

total_passages = len(df)
unique_days = df["date"].nunique()
first_ts = df["timestamp"].min()
last_ts = df["timestamp"].max()
avg_per_day = total_passages / unique_days if unique_days else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de passagens", total_passages)
c2.metric("Dias com registro", unique_days)
c3.metric("Média por dia", f"{avg_per_day:.2f}")
c4.metric("Período coletado", f"{first_ts.date()} → {last_ts.date()}")

# ============================================================
# INTERPRETAÇÃO GERAL
# ============================================================

st.markdown("## 📘 Interpretação Inicial dos Dados")

st.markdown(f"""
- **Total de {total_passages} passagens** registradas no período analisado.  
- Registro distribuído ao longo de **{unique_days} dias distintos**.  
- Coleta começou em **{first_ts}** e terminou em **{last_ts}**.  
- Média aproximada de **{avg_per_day:.2f} passagens por dia**.  
Esses valores ajudam a dimensionar a atividade da portaria.
""")

# ============================================================
# SÉRIE TEMPORAL POR HORA
# ============================================================

st.header("⏱ Série Temporal Geral")

ts_hour = df.groupby(df["timestamp"].dt.floor("H")).size().reset_index(name="count")
fig_ts = px.line(ts_hour, x="timestamp", y="count", title="Passagens por hora")
st.plotly_chart(fig_ts, use_container_width=True)

st.markdown("""
### 🧠 Interpretação
A série temporal evidencia os períodos de maior fluxo.
Picos podem estar associados a:
- Entrada/saída de moradores  
- Período de entregas  
- Visitas  
- Troca de turno  

Quedas longas podem indicar ausência de fluxo ou falha na coleta.
""")

# ============================================================
# HEATMAP
# ============================================================

st.header("🔥 Mapa de Calor — Dia da Semana × Hora")

heat = df.groupby(["weekday_num", "hour"]).size().reset_index(name="count")
heat_pivot = heat.pivot(index="weekday_num", columns="hour", values="count").fillna(0)

fig_heat = go.Figure(
    data=go.Heatmap(
        z=heat_pivot.values,
        x=heat_pivot.columns,
        y=heat_pivot.index,
        coloraxis="coloraxis"
    )
)

fig_heat.update_layout(
    title="Intensidade de passagens por horário e dia da semana",
    xaxis_title="Hora do dia",
    yaxis_title="Dia da semana (0=Segunda)"
)

fig_heat.update_layout(coloraxis={'colorscale':'Viridis'})

st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("""
### 🧠 Interpretação do Heatmap

- Blocos escuros representam **horários com maior fluxo**.  
- Blocos claros indicam **períodos de baixa circulação**.  
- É possível observar padrões característicos de movimentação diária.  
""")

# ============================================================
# INTERARRIVAL
# ============================================================

st.header("⏳ Tempo Entre Passagens (Interarrival)")

inter = df["delta_s"].dropna()
fig_inter = px.histogram(inter, nbins=50, title="Histograma de tempo entre eventos (segundos)")
fig_inter.update_layout(xaxis_title="Segundos", yaxis_title="Frequência")

st.plotly_chart(fig_inter, use_container_width=True)

st.markdown("""
### 🧠 Interpretação do Interarrival

- Valores **muito baixos** (< 1 segundo) podem indicar **leituras duplicadas**.  
- Intervalos longos podem indicar períodos de calma ou falhas temporárias.  
- Ajuda a avaliar a qualidade da leitura do sensor.  
""")

# ============================================================
# PREVISÃO BASELINE
# ============================================================

st.header("📈 Previsão Baseline (Média por Dia da Semana × Hora)")

hourly = df.copy()
hourly["weekday_num"] = hourly["timestamp"].dt.weekday
hourly["hour"] = hourly["timestamp"].dt.hour
mean_wdhr = hourly.groupby(["weekday_num","hour"]).size().reset_index(name="mean_count")

next_day = (df["timestamp"].max() + pd.Timedelta(days=1)).date()
next_wd = next_day.weekday()

pred = []

for h in range(24):
    match = mean_wdhr[(mean_wdhr["weekday_num"] == next_wd) & (mean_wdhr["hour"] == h)]
    mu = float(match["mean_count"].iloc[0]) if not match.empty else 0.0
    sigma = np.sqrt(mu)
    lower = max(0, mu - 1.96 * sigma)
    upper = mu + 1.96 * sigma
    pred.append([h, mu, lower, upper])

pred_df = pd.DataFrame(pred, columns=["hora", "previsao", "ic_inf", "ic_sup"])

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=pred_df["hora"], y=pred_df["previsao"], mode="lines+markers", name="Previsão"))
fig_pred.add_trace(go.Scatter(x=pred_df["hora"], y=pred_df["ic_inf"], mode="lines", name="IC 95% inferior"))
fig_pred.add_trace(go.Scatter(x=pred_df["hora"], y=pred_df["ic_sup"], mode="lines", name="IC 95% superior"))

fig_pred.update_layout(
    title=f"Previsão para {next_day} (weekday={next_wd})",
    xaxis_title="Hora do dia",
    yaxis_title="Contagem esperada"
)

st.plotly_chart(fig_pred, use_container_width=True)

st.markdown("""
### 🧠 Interpretação da Previsão

A previsão é baseada na **média histórica** da mesma combinação
**(dia da semana × hora)**.

- Funciona bem mesmo com poucos dados.  
- Dá uma ideia dos horários onde se espera **maior ou menor fluxo** amanhã.  
- O intervalo de confiança 95% (IC) mostra a incerteza.  
""")

st.dataframe(pred_df, height=300)
