# Importando as bibliotecas necessárias para a análise
import pandas as pd
import numpy as np # Importa a biblioteca numpy, essencial para operações numéricas e arrays.
import re # Biblioteca para trabalhar com expressões regulares, usada para a limpeza dos nomes de colunas.
import plotly.express as px # Biblioteca para criar gráficos interativos de forma fácil e rápida.
import plotly.graph_objects as go # Biblioteca mais avançada para criar gráficos mais complexos e personalizados.

# Importando bibliotecas para análise de séries temporais
from statsmodels.tsa.seasonal import seasonal_decompose # Importa a função para decompor uma série temporal em tendência, sazonalidade e resíduo.
import matplotlib.pyplot as plt # Usada para a visualização dos componentes da série temporal.

# Instalação das bibliotecas necessárias para os modelos de previsão
# O comando !pip instala a biblioteca 'pmdarima' para o modelo ARIMA
!pip install pmdarima
# O comando !conda instala a biblioteca 'fbprophet' para o modelo de crescimento
# -c conda-forge especifica o canal de instalação
!conda install fbprophet -c conda-forge

# --- Preparação e Limpeza dos Dados ---

# As próximas linhas assumem que você já carregou os dados em um DataFrame chamado 'df'.
# O seu código original não mostra o carregamento, mas a lógica de limpeza é a seguinte:

# Conferindo os tipos de cada coluna para verificar se estão corretos.
df.dtypes

# Função para padronizar os nomes das colunas.
# remove caracteres especiais e espaços, convertendo para minúsculas.
def corrige_colunas(col_name):
  return re.sub(r"[/| ]", "", col_name).lower()

# Aplica a função de limpeza a todas as colunas do DataFrame.
df.columns = [corrige_colunas(col) for col in df.columns]
df # Exibe o DataFrame com os nomes das colunas corrigidos.

# Selecionando apenas os dados referentes ao Brasil.
# 'df.loc[]' é usado para selecionar linhas com base em uma condição.
df.loc[df.countryregion =="Brazil"]

# Filtrando os dados do Brasil onde o número de casos confirmados é maior que zero.
brasil = df.loc[
    (df.countryregion =="Brazil") &
    (df.confirmed > 0)
]
brasil # Exibe o DataFrame 'brasil' filtrado.

# --- Análise e Visualização dos Dados ---

# Gráfico de linha interativo com a evolução dos casos confirmados.
# 'px.line' do Plotly Express cria um gráfico de linha.
# Eixo X: data de observação. Eixo Y: casos confirmados.
px.line(brasil, 'observationdate', 'confirmed', title='Casos confirmados no Brasil')

# Calculando novos casos por dia usando programação funcional (map e lambda).
# 'map' aplica uma função (lambda) a cada elemento de uma sequência.
# A função lambda calcula a diferença entre o número de casos do dia atual e o dia anterior.
brasil['novoscasos'] = list(map(
    lambda x: 0 if (x==0) else brasil['confirmed'].iloc[x] - brasil['confirmed'].iloc[x-1],
    np.arange(brasil.shape[0]) # np.arange cria uma sequência de números de 0 até o número de linhas do DataFrame.
))

brasil # Exibe o DataFrame com a nova coluna 'novoscasos'.

# Visualizando o gráfico de novos casos por dia.
px.line(brasil, x='observationdate', y='novoscasos', title='Novos casos por dia')

# Gráfico de mortes usando 'go.Figure', que oferece mais controle sobre os elementos do gráfico.
fig = go.Figure()
# Adiciona um "traço" (trace) para o gráfico de mortes.
# 'go.Scatter' cria um gráfico de linha com marcadores nos pontos de dados.
fig.add_trace(
    go.Scatter(x=brasil.observationdate, y=brasil.deaths, name='Mortes', mode='lines+markers',
               line=dict(color='red'))
)
# Atualiza o layout do gráfico, adicionando título aos eixos e ao gráfico.
fig.update_layout(title='Mortes por COVID-19 no Brasil',
                    xaxis_title='Data',
                    yaxis_title='Total de Mortes')
fig.show() # Exibe o gráfico.

# --- Funções para Taxa de Crescimento ---

# Função para calcular a taxa de crescimento média.
def taxa_crescimento(data, variable, data_inicio=None, data_fim=None):
  # Se a data de início não for especificada, encontra a primeira data com casos > 0.
  if data_inicio == None:
    data_inicio = data.observationdate.loc[data[variable] > 0].min()
  else:
    data_inicio = pd.to_datetime(data_inicio)
    
  # Se a data de fim não for especificada, usa a última data disponível.
  if data_fim == None:
    data_fim = data.observationdate.iloc[-1]
  else:
    data_fim = pd.to_datetime(data_fim)

  # Encontra os valores de "passado" e "presente" com base nas datas.
  passado = data.loc[data.observationdate == data_inicio, variable].values[0]
  presente = data.loc[data.observationdate == data_fim, variable].values[0]

  # Calcula o número de dias entre as datas.
  n = (data_fim - data_inicio).days

  # Calcula a taxa de crescimento e converte para porcentagem.
  taxa = (presente / passado)**(1/n) - 1

  return taxa*100

# Calcula a taxa de crescimento média do COVID no Brasil em todo o período.
taxa_crescimento(brasil, 'confirmed')

# Função para calcular a taxa de crescimento diária.
def taxa_crescimento_diaria(data, variable, data_inicio=None):
  # Se a data de início for None, define como a primeira data disponível com casos > 0.
  if data_inicio == None:
    data_inicio = data.observationdate.loc[data[variable] > 0].min()
  else:
    data_inicio = pd.to_datetime(data_inicio)

  data_fim = data.observationdate.max()
  # Define o número de pontos no tempo que vamos avaliar.
  n = (data_fim - data_inicio).days

  # Calcula a taxa de crescimento de um dia para o outro usando 'map' e 'lambda'.
  taxas = list(map(
      lambda x: (data[variable].iloc[x] - data[variable].iloc[x-1]) / data[variable].iloc[x-1],
      range(1, n+1)
  ))
  return np.array(taxas) * 100

# Executa a função para calcular a taxa de crescimento diária.
tx_dia = taxa_crescimento_diaria(brasil, 'confirmed')

# A linha 'tx_dia' no código original está solta. A linha 'tix_dia =...' já faz essa chamada.
# 'tx_dia' é a variável que armazena o resultado.
# O print ou o gráfico abaixo seria o ideal para mostrar o resultado.

# Define o primeiro dia com casos confirmados para usar no gráfico.
primeiro_dia = brasil.observationdate.loc[brasil.confirmed > 0].min()

# Gráfico da taxa de crescimento diária.
px.line(x=pd.date_range(primeiro_dia, brasil.observationdate.max())[1:],
        y=tx_dia, title='Taxa de crescimento de casos confirmados no Brasil')

# --- Decomposição da Série Temporal ---

# A linha de importação já foi corrigida no início do script para evitar o erro.
# from statsmodels.tsa.seasonal import seasonal_decompose # O módulo é 'statsmodels' e o sub-módulo é 'seasonal'
# import matplotlib.pyplot as plt

# A função 'seasonal_decompose' divide a série temporal em componentes.
# 'confirmados' precisa ser uma série com índice de data/hora.
res = seasonal_decompose(confirmados)

# Cria 4 subplots para visualizar cada componente da decomposição.
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,8))

# Plota os componentes em cada subplot.
ax1.plot(res.observed, label='Observado') # A série original.
ax2.plot(res.trend, label='Tendência') # A tendência de longo prazo.
ax3.plot(res.seasonal, label='Sazonalidade') # O padrão que se repete em intervalos regulares.
ax4.plot(confirmados.index, res.resid, label='Resíduo') # O que sobra após remover a tendência e a sazonalidade.
ax4.axhline(0, linestyle='dashed', c='green') # Adiciona uma linha horizontal para indicar o resíduo zero.
plt.show() # Exibe o gráfico.

# --- Modelo ARIMA para Previsão ---

# Instalação do 'pmdarima', uma biblioteca que automatiza a escolha dos melhores parâmetros para o modelo ARIMA.
# O comando '!pip install pmdarima' foi adicionado no início.

# 'auto_arima' encontra automaticamente os melhores parâmetros (p, d, q) para o modelo.
modelo = auto_arima(confirmados)

# Criação do gráfico com as previsões do modelo ARIMA usando go.Figure.
fig = go.Figure(go.Scatter(
    x=confirmados.index, y=confirmados, name='Observados' # Dados reais.
))
fig.add_trace(go.Scatter(
    x=confirmados.index, y=modelo.predict_in_sample(), name='Preditos' # Valores previstos pelo modelo para os dados já existentes (fit).
))
fig.add_trace(go.Scatter(
    x=pd.date_range('2020-05-20', '2020-06-20'), y=modelo.predict(31), name='Forecast' # Previsão para os próximos 31 dias.
))
fig.update_layout(title='Previsão de casos confirmados no Brasil para os próximos 30 dias')
fig.show()

# --- Modelo de Crescimento (Prophet) ---

# Instalação do fbprophet, biblioteca do Facebook para previsões de séries temporais.
# O comando '!conda install fbprophet -c conda-forge' foi adicionado no início.

# Pre-processamento para o Prophet.
# O Prophet exige que as colunas se chamem 'ds' (data) e 'y' (valor).
# Os últimos 5 dias são separados para o conjunto de teste.
train = confirmados.reset_index()[:-5]
test = confirmados.reset_index()[-5:]

# Renomeando as colunas conforme o requisito do Prophet.
train.rename(columns={"observationdate":"ds", "confirmed":"y"}, inplace=True)
test.rename(columns={"observationdate":"ds", "confirmed":"y"}, inplace=True)

# Define o modelo de crescimento logístico, que simula um teto (capacidade máxima).
# 'changepoints' são pontos onde a tendência da série pode mudar.
profeta = Prophet(growth="logistic", changepoints=['2020-03-21', '2020-03-30', '2020-04-25', '2020-05-03', '2020-05-10'])
pop = 211463256 # Define o teto da capacidade, neste caso a população do Brasil.
train['cap'] = pop # Adiciona a coluna 'cap' ao DataFrame de treino.

# Treina o modelo com os dados de treino.
profeta.fit(train)

# Constrói um DataFrame com datas futuras para fazer a previsão.
# 'periods=200' cria 200 novas datas no futuro.
future_dates = profeta.make_future_dataframe(periods=200)
future_dates['cap'] = pop # Adiciona a coluna 'cap' também às datas futuras.
forecast = profeta.predict(future_dates) # Gera a previsão.

# Criação do gráfico final para visualizar o modelo Prophet.
fig = go.Figure()
# Plota a linha de previsão ('yhat' é o nome da coluna de previsão do Prophet).
fig.add_trace(go.Scatter(x=forecast.ds, y=forecast.yhat, name='Predição'))
# Plota os dados reais que foram usados para o treino.
fig.add_trace(go.Scatter(x=train.ds, y=train.y, name='Observados - Treino'))
fig.update_layout(title='Predições de casos confirmados no Brasil')
fig.show() # Exibe o gráfico final.
