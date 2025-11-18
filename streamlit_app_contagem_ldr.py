# Streamlit app: Contagem de Passagens (LDR)
# Salve este arquivo como `streamlit_app_contagem_ldr.py` e rode:
# pip install streamlit pandas plotly numpy
# streamlit run streamlit_app_contagem_ldr.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title='Dashboard - Contagem LDR', layout='wide')

st.title('📡 Dashboard de Contagem de Passagens — LDR')
st.markdown('''
Este dashboard permite explorar os dados de passagens detectadas por um sensor LDR (feixe interrompido).
Carregue um CSV com colunas: `id`, `passagens` (número da passagem) e `timestamp` (data e hora).
''')

# Sidebar - upload / opções
st.sidebar.header('Carregamento e filtros')
uploaded = st.sidebar.file_uploader('Carregar CSV (ou use o arquivo padrão se disponível)', type=['csv'])
use_default = False
if uploaded is None:
    try:
        df = pd.read_csv('/mnt/data/contagem_ldr_202511181441.csv')
        use_default = True
    except Exception:
        df = None
else:
    df = pd.read_csv(uploaded)

if df is None:
    st.warning('Nenhum arquivo CSV carregado e arquivo padrão não encontrado. Faça upload do seu CSV.')
    st.stop()

# Basic cleaning
# Normalizar nomes de coluna
df = df.rename(columns=lambda x: x.strip().lower())
expected = ['id', 'passagens', 'timestamp']
if not set(expected).issubset(set(df.columns)):
    st.error(f'As colunas esperadas não foram encontradas. Encontrei: {list(df.columns)}.\nEspere as colunas: {expected}')
    st.stop()

# Parse timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
missing_ts = df['timestamp'].isna().sum()
if missing_ts>0:
    st.warning(f'Tem {missing_ts} timestamps inválidos que serão removidos.')
    df = df.dropna(subset=['timestamp']).copy()

# Ordenar por id e timestamp para consistência
df = df.sort_values(['id','timestamp']).reset_index(drop=True)

# Criar colunas auxiliares
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.floor('min')

df['weekday'] = df['timestamp'].dt.day_name()

# Inter-arrival time (segundos) - por sequência contínua de id
# Considerando que id identifica uma sequência contínua de leituras (o usuário explicou isso)
# Vamos agrupar por id e calcular diff

def compute_interarrival(df):
    df = df.sort_values('timestamp')
    df['prev_ts'] = df['timestamp'].shift(1)
    df['delta_s'] = (df['timestamp'] - df['prev_ts']).dt.total_seconds()
    # Para que o primeiro evento de cada 'id' não gere um delta gigante, setar NaN
    df.loc[df['id'] != df['id'].shift(1), 'delta_s'] = np.nan
    return df

df = df.groupby('id', group_keys=False).apply(compute_interarrival).reset_index(drop=True)

# Sidebar filters
min_date = df['date'].min()
max_date = df['date'].max()
date_range = st.sidebar.date_input('Intervalo de datas', [min_date, max_date])
if len(date_range)==2:
    start_date, end_date = date_range
else:
    start_date, end_date = date_range[0], date_range[-1]

mask = (df['date']>=start_date) & (df['date']<=end_date)
filtered = df.loc[mask].copy()

# Agregação por unidade de tempo
aggregation = st.sidebar.selectbox('Agregação temporal para série', ['minuto','hora','dia'])
if aggregation=='minuto':
    ts = filtered.groupby('minute').size().rename('count').reset_index()
    ts['x'] = ts['minute']
elif aggregation=='hora':
    ts = filtered.groupby([filtered['timestamp'].dt.floor('H')]).size().rename('count').reset_index()
    ts['x'] = ts['timestamp']
else:
    ts = filtered.groupby(filtered['timestamp'].dt.floor('D')).size().rename('count').reset_index()
    ts['x'] = ts['timestamp']

# KPIs
total_passages = len(filtered)
unique_days = filtered['date'].nunique()
avg_per_day = total_passages/unique_days if unique_days else 0
first_ts = filtered['timestamp'].min()
last_ts = filtered['timestamp'].max()
peak_hour = int(filtered.groupby('hour').size().idxmax()) if not filtered.empty else None

col1, col2, col3, col4 = st.columns(4)
col1.metric('Total de passagens', f'{total_passages}')
col2.metric('Dias monitorados (no filtro)', f'{unique_days}')
col3.metric('Média por dia', f'{avg_per_day:.2f}')
col4.metric('Hora de pico (hora do dia)', f'{peak_hour if peak_hour is not None else "-"}')

# Time series
st.subheader('Série temporal de passagens')
fig_ts = px.line(ts, x='x', y='count', title='Passagens ao longo do tempo (agregação selecionada)')
fig_ts.update_layout(hovermode='x unified')
st.plotly_chart(fig_ts, use_container_width=True)

# Cumulative
st.subheader('Contagem acumulada')
cum = filtered.sort_values('timestamp').reset_index(drop=True)
cum['cumcount'] = np.arange(1, len(cum)+1)
fig_cum = px.line(cum, x='timestamp', y='cumcount', title='Contagem acumulada de passagens')
st.plotly_chart(fig_cum, use_container_width=True)

# Heatmap hour x weekday
st.subheader('Mapa de calor: horário x dia da semana')
heat = filtered.groupby(['weekday','hour']).size().rename('count').reset_index()
# Garantir ordem dos dias
weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
heat['weekday'] = pd.Categorical(heat['weekday'], categories=weekdays, ordered=True)
heat = heat.sort_values(['weekday','hour'])
heat_pivot = heat.pivot_table(index='weekday', columns='hour', values='count', fill_value=0)
fig_heat = go.Figure(data=go.Heatmap(z=heat_pivot.values, x=heat_pivot.columns, y=heat_pivot.index))
fig_heat.update_layout(title='Contagens por hora e dia da semana')
st.plotly_chart(fig_heat, use_container_width=True)

# Interarrival times
st.subheader('Tempo entre passagens (interarrival)')
if filtered['delta_s'].dropna().empty:
    st.write('Não há interarrival disponíveis para o filtro selecionado (poucos eventos ou falta de sequência).')
else:
    inter = filtered['delta_s'].dropna()
    fig_hist = px.histogram(inter, nbins=50, labels={'value':'Segundos'}, title='Histograma de tempo entre eventos (s)')
    st.plotly_chart(fig_hist, use_container_width=True)
    st.write('Resumo do interarrival (s):')
    st.write(inter.describe())

# Anomalias e alertas simples
st.subheader('Alertas automáticos (checagem rápida)')
alerts = []
# Gaps grandes: maior que média + 3*std
if not filtered['delta_s'].dropna().empty:
    mean = filtered['delta_s'].mean()
    std = filtered['delta_s'].std()
    threshold = mean + 3*std
    large_gaps = filtered[filtered['delta_s']>threshold]
    if not large_gaps.empty:
        alerts.append(f'{len(large_gaps)} intervalo(s) entre eventos muito grande(s) (>{threshold:.0f}s).')

# Bursts: janelas de 1 minuto com contagem > média por minuto * 4 (exemplo)
per_min = filtered.groupby(filtered['timestamp'].dt.floor('T')).size()
if not per_min.empty:
    thr_burst = per_min.mean()*4
    bursts = per_min[per_min>thr_burst]
    if not bursts.empty:
        alerts.append(f'{len(bursts)} minuto(s) com atividade muito acima da média (picos).')

if alerts:
    for a in alerts:
        st.warning(a)
else:
    st.success('Nenhum alerta automático detectado para o filtro atual.')

# Tabela de dados com opção de download
st.subheader('Dados brutos (filtrados)')
st.dataframe(filtered[['id','passagens','timestamp','delta_s']].sort_values('timestamp'), height=300)

@st.cache_data
def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = to_csv(filtered)
st.download_button('Baixar CSV filtrado', data=csv, file_name='contagem_ldr_filtrado.csv', mime='text/csv')

# Seção de interpretações automáticas (texto)
st.subheader('Interpretações rápidas (automáticas)')
st.markdown('''
- O KPI "Total de passagens" mostra quantas vezes o feixe foi interrompido no período selecionado.
- "Hora de pico" indica a hora do dia com maior número de passagens (útil para identificar fluxo de pessoas).
- O mapa de calor ajuda a identificar dias da semana e horários com mais movimento (ex.: entrada/saída de turno).
- Interarrival muito pequenos em sequência podem indicar leitura duplicada ou ruído; intervalos muito grandes podem indicar falta de monitoramento ou falha momentânea.
''')

st.markdown('---')
st.markdown('**Observação:** personalize os thresholds de detecção de anomalias no código conforme o comportamento real do seu sensor.')

st.markdown('Se quiser que eu adapte o layout, adicione gráficos específicos (ex.: gráfico por apartamento/porta), ou gere um relatório em LaTeX/Word com as interpretações, me diga o que quer mudar.')
