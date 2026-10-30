### Tipos de dados

Muitas vezes o tipo de dados não é o que estou a ver, mas sim o significado que lhe quero atribuir.

- Numéricos - discretos e contínuos
- Categóricos (nominal) - ou acertei ou não acertei, os error sao binários.
- Ordinal - os error são discretos

### Média, Mediana e Moda

Média é muito influenciada pelos outliers: normalmente temos de identificar se temos muitos outlier e damos menos valor à média e valorizamos mais a mediana.

### Variância e Desvio Padrão

Mede a dispersão.

A variância quer se grande, para podermos extrair informação.

Variância é o quadrado do que estamos a medir.

Se queremos trabalhar na mesma escala dos dados temos de fazer a raiz quadrada, o que resulta no desvio padrão.

(Calculo da variância ver slide.)

### Distribuição Normal

Em muitos algoritmos é necessario que os dados se encontrem numa distribuição normal.

Entre -1 e 1 desvio padrão temos 34.1% dos nossos casos,... (ver slide).

Também é preciso fazer para variáveis categóricas.

### Percentis

Boa ferramenta para visualizar a distribuição dos dados. 50% corresponde à mediana.

### Quartis

A àrea entre o quartil 1 e 3 corresponde ao local onde 50% dos dados caem.

### Covariância

Utiliza a `média` de duas variáveis para avaliar se as duas variáveis variam de forma sistemática entre elas.

Exemplo: data de nascimento e idade têm covariância direta. quando uma é maior a outra é maior de igual forma.

É a média das variâncias

Covariância aplica-se quando temos a população toda.

Na maior parte das vezes utilizamos a `sample covariance` (ver slide).

Não está na escala uma vez que é calculada a partir da `variância`.
Isto é: dá para perceber se existe covariância, mas para perceber o grau temos de utilizar a `Correlação`

### Correlação

Calculada a partir da `covariância`.

Correlação não implica `causalidade`.

Exemplo da correlação entre gelados e crime em NYC. Mas neste caso podemos encontrar um terceiro atributo que tenha correlação e causalidade com ambos os atributos anterior: temperatura. Quando está melhor tempo as pessoas comem mais gelados e os criminosos cometem mais crimes.

### Visualização

O histograma tem de aparecer sempre.

### Ferramentas

Nao ao Anaconda sim ao miniconda.

## T2

### Aprendizagem simbólica vs não simbólica

Um humano consegue interpretar/perceber os dados ou não.

Exemplo: LLMs são não simbólicos porque não conseguimos interpretar as decisões.

### Aprendizagem supervisionada e não supervisionada

Na não supervisionada é mais difícil interpretar os resultados do modelo, uma vez que não há um mapeamento direto para variáveis independentes (y).

### Aprendizgem por reforço (RL)

Não supervisionado, mas tem caracteristicas de supervisionado.

### Manipulação de dados

Ter em atenção:

- Missing values
- Duplicated values
- Noise (quando temos uma área em vez de um linha)
- Outliers

Para detetar problemas:

- Central tendency
- Statistical dispersion
- Probability distribution
- Correlation/Dependence
- Contingency tables (Correlação para dados categóricas)
- Data vizualization

Redução dimencional é muito importante para data viz.

## Observações

O prof acredita que o futuro é a aprendizagem não supervisionada e RL. Porque os dados são caros e menos flexiveis.

Exemplo da condução autónoma; abordagens:
- fazer labeling de dados de videos à mão (waymoo)
- fazer como a tesla que tem os carros que vende a gerar dados e a fazer auto labeling de acordo com a condução e os seus outcomes.

Modelos têm um prazo de validade. Ao fim de algum tempo o modelo atrofia/degrada.

A metodologia preferida do professor é 
- conhecer o domínio
- ler os dados
- preparar os dados
- segragação/split dos dados
- fit do modelo
- tunning
- deployment
- monitorização contínua.

A maior parte das pessoas despreza o primeiro passo que é conhecer o domínio.

Exemplo dos aviões que voltavam com furos de balas: temos de proteger onde esses aviões não têm furos, porque eles voltaram por não terem furos nesses sítios.

Antes de tratar os dados a fundo devemos estabelecer uma baseline end-to-end.

O mais importante asseguir a conhecer o domínio é a qualidade dos dados, porque requer interpretação e sentido crítico.

No trabalho temos de ter a capacidade de perceber se o nosso modelo é ou não genérico.

Ensemble learning é uma das melhores técnicas para o trabalho, porque são dados tablados.
Random forest também. Usa bagging 

### Preparação avançada dos dados

Escala dos dados - se o algoritmo não for sensível à escala como a decision tree não é preciso
- normalização - entre 0 e 1 (preferível quando não temos viés na distribuição normal)
- standardização - usa a media e desvio padrão (preferível quando temos viés na distribuição normal)

Deteção de outliers - detetar atrvés de vizualização: box plot, wisker plot
- drop
- cap

Seleção de atributos - alguns algoritmos são muito sensíveis à quantidade de atributos
- remover colunas invariantes
- remover colunas com pouca variância
- remover colunas altamente correlacionadas
- PCA (ver slide)
- Wrapper methods (ver slide) - tentativa erro 
- Embedded methods (ver slide)

Dados em falta
- remover
- usar a média
- usar a mediana
- interpolation?
- masking?

Discretização de valores nominais
- one-hot encoding - passar para encoding booleano (0 ou 1) aumentando o número de atributos
- label encoding -
- binary encoding - passar para notação binária 7 -> 111
- binning - empactorar intervalos de valores. util quando valores contínuos descrevem a mesma coisa no contexto do problema, exemplo: saude de alguem com 20 ou 21 anos. remove o ruido

Feature engineering

Ter cuidado com a ordenação dos datasets antes de fazer os split!

Analize de uma distribuição normal: skewness e kurtosis perto de 0
kurtosis negativo: montanha esguia
skewness positivo: assimetria à esquerda

pair plot = super power

com threasholds podemos criar algoritmos que não chegam ao modelo, se uma classe pode ser derivada a partir de um atributo, podemos simplificar o nosso modelo.

### Modelos lineares

é necessário verficar a linearidade

definição de linearidade: $f(x + y) = f(x) f(y) \wedge  f(ax) = af(x)$

classificação: uma linha separa as classes
regressão: valores perto de uma linha

NOTA: dispercei no slide antes de "Multiple linear regression"

regressão logística - utilizado para classificação binária.

e se quisermos utilizar regressão logistica para várias classes? multiplo as classes e percebo qual é a que tem maior probabilidade, uso essa. TODO: rever com o chat

estimar parametros: método dos mínimos quadrados; lento com muitos atributos
para estimar parametros quando o método anterior não é computável: gradient descend; é preciso encontrar o learning rate apropriado

batch gradient descent: lento para muitas linhas
stocastic e mini-batchgradient descent: sempre rápido.

Regularização:
lasso regression - automáticamente seleciona atributos quando não são relevantes
ridge regression - parecido com a anterior, mas NUNCA coloca os atributos a zero

### Árvores de decisão

as àrvores de decisão são um bom algoritmo para fazer baselines

é insensível a grande parte das anomalias nos dados

na classificação é preciso ter algum cuidado com os hiperparametros

os nós da àrvore estão organizados pelos atributos mais importantes no topo

impuridade é relacionada com a percentagem da discriminatoriabilidade de cada decisão

quanto menor for a impuridade mais importante é o atributo

o algoritmo é pesado

os dados influenciam diretamente a estrutura de dados, ao acrescentar dados a àrvore pode mudar. é um algoritmo instável

depth infinita é equivalente a overfitting dos dados

é preciso determinar critérios de paragem:
- cardinalidade máxima por nó
- não continuar se os próximos nós pioram o modelo
- pre-prunning:
    - min_sample_split - limitar o número de casos para as divisões
    - min_sample_leaf - limitar o número de casos para as folhas
- post-prunning: permite tomar sempre a melhor decisão de prunning.
    - ecolher uma profundidade arbitrária (exemplo da poda da cebe, nem sempre é bom)
    - cost-complexity-prunning - poda ramo a ramo. temos de descobrir o $\alpha$ ideal

###
é bom fazer cross validation contra o conjunto de testes para estudara a establidade do modelo: comparar a media com o desvio padrao

#### Ensemble learning

utiliza um conjunto de modelos

podemos colocar os modelos em série, em paralelo, ou em ?

- bagging - bootstrap aggregation - treinar modelos em subsets de dados (ex 70%) - funciona em paralelo
- boosing - funciona em série - faz boost ao peso dos erros dos casos falhados 
    - ada (adaptative) - altera dados para dar importancia as predicoes
    - Xg - altera os hiperparametros do modelo em vez de mexer nos dados
- stacking - colocar o output de um modelo a ser o input de outro
    - blending - evita utilização repetida de dados por modelos em série no treino

bagging e boosting funciona na base de àrvores, enquanto o stacking é mais orientado para mistorar diferentes tipos de modelo.

- voting - pode ser hard (democrático) ou soft (peso pela confiança que o modelo dá)

tpc: perceber o que é n_estimators

## Redes Neuronais

generalizam bem
ignoream ruido e inconsistencias nos dados de treino

funcao de ativacao determina o output: binário usa-se sigmoid, regressao usa-se linear ou RelU para valores não negativos

softmax - normalização de um conjunto de probabilidades entre 0 e 1, bom para problemas multiclasse

dá para fazer muita coisa com arquiteturas, é um mundo

- vanilla neural network - uma camada intermédia

- feed forward - os outputs não voltam para trás como input, apenas na direção do nodos finais

workflow tipico: feed forward into backpropagation

problema de backprop: vanishing gradients. o processo de backpropagation pode degenerar


