# Importa a biblioteca Streamlit para construir a interface web
import streamlit as st
# Importa o Pandas para manipulação dos dados históricos
import pandas as pd
# Importa o Numpy para operações matemáticas rápidas
import numpy as np
# Importa o Altair para construir gráficos modernos e interativos
import altair as alt
# Importa o scikit-learn para treinar o modelo e fazer previsões em tempo real
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Configura o layout do Streamlit para o modo amplo (wide)
st.set_page_config(layout="wide")

# Adiciona o título principal da aplicação solicitado
st.title("Previsão de Aluguéis")

# Carrega e limpa a base de dados histórica para treinar o modelo dinamicamente
dados_originais = pd.read_excel('aluguel-de-bicicletas.xlsx')
dados = dados_originais.drop_duplicates().dropna()

# Mapeia as variáveis categóricas de texto para valores numéricos
mapeamento_sazonalidade = {'Verão': 0, 'Outono': 1, 'Inverno': 2, 'Primavera': 3}
mapeamento_clima = {'Neblina': 0, 'Tempo claro': 1, 'Chuva forte': 2, 'Chuva leve': 3}
mapeamento_dia_semana = {'Segunda-feira': 0, 'Terça-feira': 1, 'Quarta-feira': 2, 'Quinta-feira': 3, 'Sexta-feira': 4, 'Sábado': 5, 'Domingo': 6}

dados['sazonalidade'] = dados['sazonalidade'].map(mapeamento_sazonalidade)
dados['clima'] = dados['clima'].map(mapeamento_clima)
dados['dia-semana'] = dados['dia-semana'].map(mapeamento_dia_semana)

# Prepara as variáveis explicativas e a variável alvo
X = dados.drop(columns=['dia', 'mês', 'ano', 'alugueis'])
Y = dados['alugueis']

# Divide em treino e teste e ajusta o modelo de Regressão Linear
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.30, random_state=42)
modelo = LinearRegression()
modelo.fit(X_treino, Y_treino)

# Cria a estrutura de duas colunas principais para a seção superior
col1, col2 = st.columns([1, 2])

# Primeira coluna: Formulário interativo com controles para entrada de dados
with col1:
    st.header("Dados Meteorológicos")
    # Cria uma caixa de seleção para a Estação do Ano (Sazonalidade)
    sazonalidade_sel = st.selectbox("Selecione a Estação:", list(mapeamento_sazonalidade.keys()))
    # Cria uma caixa de seleção para as Condições de Clima
    clima_sel = st.selectbox("Selecione o Clima:", list(mapeamento_clima.keys()))
    # Cria uma caixa de seleção para o Dia da Semana
    dia_semana_sel = st.selectbox("Selecione o Dia da Semana:", list(mapeamento_dia_semana.keys()))

    # Cria controles deslizantes (sliders) para as variáveis numéricas continuas usando os nomes reais das colunas
    temperatura_sel = st.slider("Temperatura (°C):", float(dados_originais['temperatura'].min()), float(dados_originais['temperatura'].max()), float(dados_originais['temperatura'].mean()))
    umidade_sel = st.slider("Umidade (%):", float(dados_originais['umidade-ar'].min()), float(dados_originais['umidade-ar'].max()), float(dados_originais['umidade-ar'].mean()))
    vento_sel = st.slider("Velocidade do Vento (km/h):", float(dados_originais['velocidade-vento'].min()), float(dados_originais['velocidade-vento'].max()), float(dados_originais['velocidade-vento'].mean()))

    # Cria seletores binários simplificados para Feriado e Dia de Trabalho
    feriado_sel = st.radio("É Feriado?", options=["Não", "Sim"])
    dia_trabalho_sel = st.radio("É Dia de Trabalho?", options=["Não", "Sim"])

# Converte as seleções do usuário para o formato numérico esperado pelo modelo
feriado_val = 1 if feriado_sel == "Sim" else 0
dia_trabalho_val = 1 if dia_trabalho_sel == "Sim" else 0
sazonalidade_val = mapeamento_sazonalidade[sazonalidade_sel]
clima_val = mapeamento_clima[clima_sel]
dia_semana_val = mapeamento_dia_semana[dia_semana_sel]

# Cria um DataFrame temporário com os dados inseridos pelo usuário
dados_usuario = pd.DataFrame([[
    sazonalidade_val, feriado_val, dia_semana_val, dia_trabalho_val,
    clima_val, temperatura_sel, umidade_sel, vento_sel
]], columns=X.columns)

# Realiza a previsão em tempo real a partir dos parâmetros do formulário
previsao_real = max(0, int(round(modelo.predict(dados_usuario)[0])))

# Filtra a base original correspondente à estação selecionada para as médias históricas
dados_estacao = dados_originais[dados_originais['sazonalidade'] == sazonalidade_sel]
media_estacao = dados_estacao['alugueis'].mean()
max_estacao = dados_estacao['alugueis'].max()
min_estacao = dados_estacao['alugueis'].min()

# Segunda coluna: Exibe os 4 cards de métricas posicionados lado a lado
with col2:
    st.header("Métricas de Previsão e Histórico")
    c1, c2, c3, c4 = st.columns(4)
    # Card 1: Exibe o valor total da previsão realizada pelo modelo
    c1.metric(label="Previsão de Aluguéis", value=f"{previsao_real} un.", delta=f"{int(round(previsao_real - media_estacao))} vs Média")
    # Card 2: Exibe a média histórica de aluguéis para a estação selecionada
    c2.metric(label="Média da Estação", value=f"{int(round(media_estacao))} un.")
    # Card 3: Exibe o máximo de aluguéis registrado naquela estação
    c3.metric(label="Máximo Registrado", value=f"{int(max_estacao)} un.")
    # Card 4: Exibe o mínimo de aluguéis registrado naquela estação
    c4.metric(label="Mínimo Registrado", value=f"{int(min_estacao)} un.")

# Cria a seção inferior para acomodar os dois gráficos modernos lado a lado
st.markdown("--- ")
graf1, graf2 = st.columns(2)

# Construção do Gráfico 1: Dispersão da temperatura com imagem da bandeira do Brasil indicando a simulação
with graf1:
    st.subheader("Dispersão de Aluguéis por Temperatura")
    # Prepara o conjunto de dados históricos em formato amigável para plotagem no Altair
    dados_plot = dados_originais.copy()

    # Cria o gráfico de dispersão com os dados históricos de temperatura e aluguéis
    pontos = alt.Chart(dados_plot).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X('temperatura:Q', title='Temperatura (°C)'),
        y=alt.Y('alugueis:Q', title='Total de Aluguéis'),
        color=alt.Color('sazonalidade:N', title='Estação')
    )

    # Cria um ponto individual representado por uma bandeira do Brasil para a simulação atual
    ponto_simulado_df = pd.DataFrame([{
        'temperatura': temperatura_sel,
        'alugueis': previsao_real,
        'url_bandeira': 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Flag_of_Brazil.svg/320px-Flag_of_Brazil.svg.png'
    }])

    ponto_simulado = alt.Chart(ponto_simulado_df).mark_image(
        width=30, height=20
    ).encode(
        x='temperatura:Q',
        y='alugueis:Q',
        url='url_bandeira:N'
    )

    # Une os gráficos sobrepondo a bandeira brasileira simulada à nuvem de pontos histórica
    graf_final1 = (pontos + ponto_simulado).properties(height=350).interactive()
    st.altair_chart(graf_final1, use_container_width=True)

# Construção do Gráfico 2: Barra comparando a simulação com as médias das estações
with graf2:
    st.subheader("Comparação com as Médias das Estações")
    # Agrupa as médias históricas de cada estação do ano para comparação direta
    medias_sazonais = dados_originais.groupby('sazonalidade')['alugueis'].mean().reset_index()
    medias_sazonais.columns = ['Categoria', 'Aluguéis']

    # Adiciona a simulação do usuário no fim da tabela para compará-la lado a lado
    linha_simulado = pd.DataFrame([['Simulado (Atual)', previsao_real]], columns=['Categoria', 'Aluguéis'])
    dados_comparativos = pd.concat([medias_sazonais, linha_simulado], ignore_index=True)

    # Cria o gráfico de barras comparativo destacando a simulação
    barras = alt.Chart(dados_comparativos).mark_bar().encode(
        x=alt.X('Categoria:N', title='Estação / Cenário', sort=None),
        y=alt.Y('Aluguéis:Q', title='Quantidade de Aluguéis'),
        color=alt.condition(
            alt.datum.Categoria == 'Simulado (Atual)',
            alt.value('#FFA500'), # Laranja vibrante para dar destaque à barra simulada
            alt.value('#1f77b4')  # Azul padrão para as médias históricas
        )
    ).properties(height=350)

    st.altair_chart(barras, use_container_width=True)
