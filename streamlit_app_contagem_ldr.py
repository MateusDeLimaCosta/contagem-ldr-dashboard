# ============================================================
# DASHBOARD LDR — VISUAL ESTILO IESB
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from sklearn.ensemble import RandomForestRegressor
import warnings

# ------------------------------------------------------------
# CONFIGURAÇÃO INICIAL
# ------------------------------------------------------------

st.set_page_config(
    page_title="Dashboard - Contagem LDR",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# BLOCO DE LIMPEZA DE AVISOS (COM PROTEÇÃO ANTI-ERRO)
# ============================================================
# 1. Ignora avisos do Python (Pandas/Sklearn)
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# CSS CUSTOMIZADO (ATUALIZADO E UNIFICADO)
# ------------------------------------------------------------

st.markdown("""
<style>
    /* Configuração Geral */
    body {
        background-color: #F5F6FA;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0A1A3B !important;
        padding: 20px;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Header Principal */
    .big-header {
        background-color: #0A1A3B;
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .big-header h1 {
        font-size: 36px;
        font-weight: 800;
        margin: 0;
        color: white;
    }
    .big-header p {
        font-size: 18px;
        font-weight: 300;
        color: #E0E0E0;
        margin-top: 10px;
    }

    /* CARTÕES DE MÉTRICAS (VISÃO GERAL) */
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border-left: 6px solid #0A1A3B; /* Faixa azul IESB */
        text-align: center;
        transition: transform 0.2s;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-5px); /* Efeito de elevação ao passar o mouse */
    }
    .metric-title {
        color: #7f8c8d;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    .metric-value {
        color: #0A1A3B;
        font-size: 42px; /* Fonte bem grande e bonita */
        font-weight: 800;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 13px;
        color: #27ae60;
        font-weight: 600;
        margin-top: 5px;
    }
    
    /* Footer */
    .footer {
        margin-top: 50px;
        text-align: center;
        color: #95a5a6;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# SIDEBAR — LOGO + FILTROS
# ------------------------------------------------------------

st.sidebar.image("logo_iesb.png", use_column_width=True)

st.sidebar.markdown("## 🔧 Filtros Gerais")

# Carregar CSV padrão diretamente
try:
    df = pd.read_csv("contagem_ldr_atualizado.csv", sep=";", engine="python")
except:
    st.error("Erro: Arquivo 'contagem_ldr_atualizado.csv' não encontrado.")
    df = pd.DataFrame()

# ------------------------------------------------------------
# PRÉ-PROCESSAMENTO
# ------------------------------------------------------------

df.columns = df.columns.str.strip().str.lower().str.replace('"', "")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.day_name()
df["weekday_num"] = df["timestamp"].dt.weekday

df["prev_ts"] = df["timestamp"].shift(1)
df["prev_id"] = df["id"].shift(1)
df["delta_s"] = (df["timestamp"] - df["prev_ts"]).dt.total_seconds()
df.loc[df["id"] != df["prev_id"], "delta_s"] = np.nan


# ------------------------------------------------------------
# FILTROS LATERAIS
# ------------------------------------------------------------

st.sidebar.markdown("### 📅 Intervalo de Datas")
min_d, max_d = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Período:", [min_d, max_d])

df = df[(df["date"] >= date_range[0]) & (df["date"] <= date_range[1])]

# ------------------------------------------------------------

st.sidebar.markdown("### ⏰ Intervalo de Horas")
hour_min, hour_max = st.sidebar.slider(
    "Selecione a faixa de horário:",
    0, 23, (0, 23)
)
df = df[(df["hour"] >= hour_min) & (df["hour"] <= hour_max)]

# ------------------------------------------------------------

st.sidebar.markdown("### 📌 Dia da Semana")
dias = st.sidebar.multiselect(
    "Filtrar dias:",
    options=df["weekday"].unique(),
    default=df["weekday"].unique()
)
df = df[df["weekday"].isin(dias)]

# ------------------------------------------------------------

st.sidebar.markdown("### 🛠 Limpeza")
remove_dups = st.sidebar.checkbox("Remover leituras duplicadas (< 1s)", value=True)
if remove_dups:
    df = df[(df["delta_s"].isna()) | (df["delta_s"] > 1)]


# =====================================================================
# HEADER CENTRAL
# =====================================================================

st.markdown("""
<div class="big-header">
    <h1 style="font-size: 40px; margin-bottom: 10px;">📡 Dashboard de Contagem LDR</h1>
    <p style="font-size: 18px; color: #E0E0E0;">Análise inteligente de fluxo, detecção de padrões e monitoramento de anomalias.</p>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# ABAS DO CONTEÚDO
# =====================================================================

tabs = st.tabs([
    "📊 Visão Geral",
    "📅 Volume Diário",
    "📈 Perfil Horário",
    "🔥 Mapa de Calor",
    "🤖 Análise Preditiva",
])


# ---------------------------------------------------------------------
# ABA 1 — VISÃO GERAL
# ---------------------------------------------------------------------

with tabs[0]:
        # --- CORREÇÃO DE COMPATIBILIDADE ---
        # Garante que a coluna 'day_name' exista, mesmo mantendo seu pré-processamento original
        if "day_name" not in df.columns:
            dias_traducao = {
                'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta',
                'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
            }
            # Cria a coluna traduzida baseada no timestamp
            df["day_name"] = df["timestamp"].dt.day_name().map(dias_traducao).fillna(df["timestamp"].dt.day_name())

        # --- CÁLCULOS ---
        total_passages = len(df)
        unique_days = df["date"].nunique()
        avg_per_day = total_passages / unique_days if unique_days else 0
        
        # Encontrar recordes (com proteção caso o df esteja vazio)
        if not df.empty:
            top_day_row = df.groupby('date').size().sort_values(ascending=False).reset_index(name='count').iloc[0]
            top_day_str = top_day_row['date'].strftime('%d/%m')
            top_day_val = top_day_row['count']
            
            top_hour_row = df.groupby('hour').size().sort_values(ascending=False).reset_index(name='count').iloc[0]
            top_hour_str = f"{top_hour_row['hour']}h"
            
            # Agora funciona porque garantimos a criação de 'day_name' ali em cima
            top_weekday = df.groupby('day_name').size().idxmax()
        else:
            top_day_str, top_day_val, top_hour_str, top_weekday = "-", 0, "-", "-"

        # --- LAYOUT DOS CARTÕES ---
        st.markdown("### 🚀 Resumo de Performance")

        # Funçãozinha interna para desenhar o cartão (fica mais limpo)
        def card(title, value, subtext=""):
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{subtext}</div>
            </div>
            """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1: card("Total de Detecções", f"{total_passages:,}".replace(",", "."), "Passagens registradas")
        with c2: card("Média Diária", f"{int(avg_per_day)}", "Pessoas/Dia")
        with c3: card("Recorde Diário", f"{top_day_val}", f"Ocorrido em {top_day_str}")

        st.markdown("<br>", unsafe_allow_html=True)

        c4, c5, c6 = st.columns(3)
        with c4: card("Dias Monitorados", f"{unique_days}", "Dias de coleta")
        with c5: card("Horário de Pico", f"{top_hour_str}", "Maior concentração")
        with c6: card("Dia + Movimentado", f"{top_weekday}", "Padrão Semanal")
        
        st.markdown("---")
        
        st.success(f"""
        **📌 Análise Executiva:**
        O sensor LDR operou por **{unique_days} dias**, registrando um total de **{total_passages} passagens**.
        O padrão de comportamento indica fluxo intenso às **{top_hour_str}**, sendo **{top_weekday}** o dia da semana tipicamente mais movimentado.
        """)


# ---------------------------------------------------------------------
# ABA 2 — SÉRIE TEMPORAL
# ---------------------------------------------------------------------

with tabs[1]:
    st.header("📅 Volume Diário de Passagens")
    
    daily_counts = df.groupby("date").size().reset_index(name="contagem")
    
    # FORMATAR A DATA COMO TEXTO SIMPLES (DD/MM)
    # Isso obriga o gráfico a colar as barras
    daily_counts["data_formatada"] = pd.to_datetime(daily_counts["date"]).dt.strftime('%d/%m')
    
    fig_daily = px.bar(
        daily_counts, 
        x="data_formatada",  # Usamos a string formatada
        y="contagem", 
        text="contagem",
        title="Total de Passagens por Dia",
        color="contagem",
        color_continuous_scale="Blues"
    )
    
    fig_daily.update_traces(textposition='outside')
    
    # O PULO DO GATO: type='category' remove os buracos de datas vazias
    fig_daily.update_xaxes(type='category', title="Data")
    fig_daily.update_yaxes(title="Quantidade")
    
    st.plotly_chart(fig_daily, use_container_width=True)


# ---------------------------------------------------------------------
# ABA 3 — HEATMAP
# ---------------------------------------------------------------------

with tabs[2]:
    st.header("📈 Perfil de Atividade por Hora")
    st.markdown("Análise acumulada para identificar a 'Rotina' do local.")

    # Agrupa apenas por hora (somando todos os dias)
    hourly_profile = df.groupby("hour").size().reset_index(name="contagem")

    fig_hourly = px.area(
        hourly_profile, 
        x="hour", 
        y="contagem", 
        markers=True,
        title="Perfil de Atividade (Soma de todos os dias)",
        color_discrete_sequence=["#FF7F50"] # Cor laranja/coral
    )
    fig_hourly.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        xaxis_title="Hora do Dia (0-23h)",
        yaxis_title="Volume Acumulado"
    )
    
    st.plotly_chart(fig_hourly, use_container_width=True)
    st.success("**Insight:** Note o pico expressivo às **17h**. Isso indica um padrão claro de saída ou atividade intensa no final da tarde.")


# ---------------------------------------------------------------------
# ABA 4 — INTERARRIVAL
# ---------------------------------------------------------------------

with tabs[3]:
    st.header("🔥 Mapa de Calor: Intensidade de Fluxo")
    st.markdown("Cruzamento entre **Dia da Semana** e **Hora** para ver gargalos.")

    # Preparar dados: Dia da semana (nome) vs Hora
    # Precisamos garantir que temos a coluna 'day_name' (se não tiver criado lá em cima no pré-processamento, o código abaixo cria rapidinho)
    df["day_name"] = df["timestamp"].dt.day_name()
    dias_traducao = {'Monday':'Segunda', 'Tuesday':'Terça', 'Wednesday':'Quarta', 'Thursday':'Quinta', 'Friday':'Sexta', 'Saturday':'Sábado', 'Sunday':'Domingo'}
    df["day_name"] = df["day_name"].map(dias_traducao).fillna(df["day_name"])

    heatmap_data = df.groupby(["day_name", "hour"]).size().reset_index(name="contagem")
    
    ordem_dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    
    fig_heat = px.density_heatmap(
        heatmap_data, 
        x="hour", 
        y="day_name", 
        z="contagem", 
        nbinsx=24, 
        category_orders={"day_name": ordem_dias},
        color_continuous_scale="Viridis",
        title="Concentração de Movimento (Dia x Hora)"
    )
    fig_heat.update_layout(xaxis_title="Hora", yaxis_title="Dia da Semana")

    st.plotly_chart(fig_heat, use_container_width=True)


# ---------------------------------------------------------------------
# ABA 5 — PREVISÃO
# ---------------------------------------------------------------------

with tabs[4]:
    st.header("🤖 Análise Preditiva: Eventos Atípicos (Dias)")
    st.markdown("""
    Comparação entre o que a **IA previu (Comportamento Normal)** e o que **Realmente Aconteceu** nos dias de pico.
    """)

    # 1. Preparar dados por DIA (não por hora)
    daily_ml = df.groupby('date').size().reset_index(name='contagem')
    daily_ml['date'] = pd.to_datetime(daily_ml['date'])
    daily_ml['dow'] = daily_ml['date'].dt.dayofweek # 0=Seg, 1=Ter...
    daily_ml['date_str'] = daily_ml['date'].dt.strftime('%d/%m (%a)')

    # 2. Separar dias "Normais" vs "Atípicos" (Terças e Quintas de alto fluxo)
    # Vamos considerar atípico qualquer dia com mais de 400 passagens (ajuste se necessário)
    limiar_atipico = 400
    
    treino = daily_ml[daily_ml['contagem'] < limiar_atipico]
    teste_atipico = daily_ml[daily_ml['contagem'] >= limiar_atipico].copy()

    if len(treino) > 0 and len(teste_atipico) > 0:
        # 3. Treinar Modelo (Random Forest) nos dias calmos
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(treino[['dow']], treino['contagem'])
        
        # 4. Prever o volume para os dias atípicos
        teste_atipico['predicao'] = rf.predict(teste_atipico[['dow']])

        # 5. Criar Gráfico Comparativo (Barras Agrupadas)
        fig_pred = go.Figure()

        # Barra Real
        fig_pred.add_trace(go.Bar(
            x=teste_atipico['date_str'],
            y=teste_atipico['contagem'],
            name='Realidade (Sensor)',
            marker_color='green',
            text=teste_atipico['contagem'],
            textposition='auto'
        ))

        # Barra Prevista
        fig_pred.add_trace(go.Bar(
            x=teste_atipico['date_str'],
            y=teste_atipico['predicao'],
            name='Previsão IA (Normal)',
            marker_color='red',
            text=teste_atipico['predicao'].astype(int),
            textposition='auto',
            opacity=0.7
        ))

        fig_pred.update_layout(
            title="Detecção de Anomalia: Real vs Esperado",
            xaxis_title="Dias Atípicos",
            yaxis_title="Total de Passagens",
            barmode='group' # Barras lado a lado
        )

        st.plotly_chart(fig_pred, use_container_width=True)

        # --- NOVO GRÁFICO: LINHA TEMPORAL (REAL vs MODELO) ---
        st.markdown("---")
        st.subheader("Visualização Temporal: A Curva da Anomalia")
        st.markdown("Aqui podemos ver a linha tracejada (o que a IA esperava) e a linha sólida (o que aconteceu).")

        # Vamos prever para TODOS os dias agora, para mostrar o histórico completo
        daily_ml['predicao_geral'] = rf.predict(daily_ml[['dow']])

        fig_line = go.Figure()

        # Linha 1: Realidade (Verde Sólido)
        fig_line.add_trace(go.Scatter(
            x=daily_ml['date_str'],
            y=daily_ml['contagem'],
            mode='lines+markers',
            name='Fluxo Real',
            line=dict(color='#2E86C1', width=4), # Azul forte
            marker=dict(size=8)
        ))

        # Linha 2: Predição (Vermelho Tracejado)
        fig_line.add_trace(go.Scatter(
            x=daily_ml['date_str'],
            y=daily_ml['predicao_geral'],
            mode='lines+markers',
            name='Modelo (Expectativa)',
            line=dict(color='#E74C3C', width=3, dash='dash'), # Vermelho tracejado
            marker=dict(symbol='x', size=8)
        ))

        fig_line.update_layout(
            title="Série Temporal: Identificando a Ruptura de Padrão",
            xaxis_title="Data",
            yaxis_title="Volume",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1)
        )

        st.plotly_chart(fig_line, use_container_width=True)

        st.warning("""
        **Interpretação:** As barras vermelhas mostram quanto fluxo a IA esperava para uma Terça ou Quinta comum.
        As barras verdes mostram o fluxo real registrado.
        
        A grande diferença entre elas prova que esses dias foram eventos fora da curva (anomalias).
        """)
    else:
        st.info("Não foram detectados dias com volume extremo (>400) para análise de anomalia neste filtro.")
# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown("""
<div class="footer">
Desenvolvido por <b>Mateus de Lima Costa</b>
</div>
""", unsafe_allow_html=True)
