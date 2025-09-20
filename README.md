# Análise e Predição de Casos de COVID-19 no Brasil
Este repositório contém um projeto de análise e predição de séries temporais, conduzido como parte de um estudo de caso da DIO (Digital Innovation One). O objetivo é demonstrar a aplicação de técnicas de ciência de dados para analisar a evolução da pandemia de COVID-19 no Brasil e realizar previsões.

As anotações e comentários presentes no código foram cuidadosamente elaborados para fornecer uma documentação clara de cada etapa do processo. Para aprimorar a qualidade e a clareza da explicação, foi utilizada uma ferramenta de inteligência artificial.

## Metodologia e Ferramentas
O projeto segue um fluxo de trabalho estruturado, utilizando as seguintes bibliotecas e abordagens:

Manipulação e Limpeza de Dados: Pandas e NumPy são empregados para o tratamento inicial dos dados, incluindo a padronização dos nomes das colunas e a filtragem das informações relevantes.

Análise Exploratória e Visualização: Gráficos interativos são criados com Plotly para visualizar a evolução de casos confirmados, novos casos diários e mortes. Funções personalizadas calculam as taxas de crescimento, oferecendo insights sobre a dinâmica da propagação.

Decomposição de Séries Temporais: A biblioteca Statsmodels é utilizada para decompor a série de casos em seus componentes (tendência, sazonalidade e resíduo), permitindo uma compreensão mais aprofundada dos padrões de dados.

Modelagem e Predição: Dois modelos de séries temporais são aplicados para realizar previsões:

ARIMA (pmdarima): Um modelo robusto que automatiza a identificação dos melhores parâmetros para a previsão.

Prophet (fbprophet): Um modelo de crescimento logístico, ideal para cenários com saturação, como o de uma epidemia.

## Bibliotecas Utilizadas
Pandas: Para manipulação e análise de dados.

NumPy: Para operações numéricas e tratamento de arrays.

Plotly Express e Plotly Graph Objects: Para a criação de visualizações de dados interativas.

Statsmodels: Para decomposição de séries temporais.

pmdarima: Para a modelagem automática do modelo ARIMA.

fbprophet: Para a criação do modelo de previsão Prophet.

Matplotlib: Para a visualização estática dos componentes da série temporal.

## Requisitos
Para executar o código, é necessário ter um ambiente Python configurado com as bibliotecas listadas na seção "Bibliotecas Utilizadas". As instalações podem ser realizadas usando pip ou conda, conforme indicado no próprio script.
