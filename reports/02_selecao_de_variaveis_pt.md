# Seleção de variáveis para `AGE_RANGE` — relatório de etapa

**Preparado em:** 2026-11-04
**Execução:** `c60f8f8`, árvore de trabalho limpa (`git_dirty: false`)
**Notebook:** `notebooks/04_feature_selection.ipynb`
**Registo da execução:** `outputs/04_feature_selection_executed.ipynb`
**Artefactos:** `outputs/selected_features.json`, `outputs/feature_selection_k_sweep.csv`,
`outputs/feature_selection_ranking.csv`, `outputs/feature_selection_ranking_deployed.csv`

---

## 1. Objetivo e público

Este é um **documento interno**. Está escrito para as pessoas que orientam este
projeto, não para a avaliação. Não é uma secção do relatório de 20 páginas e não
foi escrito a pensar numa grelha de correção. O seu papel é permitir que um leitor
que conhece a matéria do curso mas não o material adicional consiga acompanhar o
que esta etapa fez, verificar que foi feita com honestidade, e decidir o que fazer
a seguir.

A etapa colocou uma única questão:

> Das 255 colunas de entrada que o modelo usa atualmente, quantas podem ser
> removidas antes de o F1 macro cair?

A resposta é que **bastam 40 colunas**: 40 das 136 colunas pré-selecionadas, em
vez das 255 entradas (as 255 entradas originais expandem-se para 1.170 colunas
depois da codificação one-hot). Essa resposta vem com três ressalvas que importam
mais do que o número em si, e são a substância deste documento:

1. A pontuação agregada esconde uma reorganização por classe (ver §5.4 e §6.1).
2. A opção com melhor pontuação foi **rejeitada por uma regra**, por uma margem
   de 0,0002.
3. Quando se contabiliza o custo de *escolher* as colunas, a estimativa honesta
   é **pior do que usar tudo**.

A secção 3 é uma visita guiada aos onze conceitos que esta etapa usa e que o curso
não ensina. Cada conceito é introduzido no ponto do método em que é necessário,
pela ordem de execução do notebook. Um leitor que siga a secção 3 consegue ler as
secções 4 a 6 sem mais enquadramento.

### Uma nota sobre as faixas etárias

Todas as interpretações de idade neste documento usam o mapeamento oficial do
Kaggle (`0 = 15–17, 1 = 18–19, …`). Os dados são ambíguos quanto à faixa 0 e a
questão está em aberto com o professor; ver `TODO.md`. Nada nesta etapa depende
do mapeamento, porque a ordenação trata os rótulos como códigos.

---

## 2. Sumário executivo

| # | Constatação | Evidência | Confiança |
|---|---|---|---|
| 1 | 40 colunas igualam a linha de base de 255 colunas **em termos agregados**, mas o efeito por classe é uma reorganização: 6 classes melhoram, 10 pioram. | 0,4519 ± 0,0104 vs. 0,4531 ± 0,0096; delta médio por classe −0,0098 | Alta |
| 2 | **K = 60 tem a melhor média e a regra rejeitou-o** por uma diferença de 0,0002. Foi a regra, e não os dados, que escolheu 40. | 0,4587, pior repetição −0,0102 contra uma guarda de −0,010 | Alta |
| 3 | A verificação da Etapa 3 compensou o seu custo: a ordenação ordinal colocava `PC1A` em 33.º, e a coluna teria sido descartada. | Spearman 0,492; `PC1A` passou da posição 33 para a 1 | Alta |
| 4 | O otimismo de seleção é **+0,0137**, e a lista final é **instável**: 26 a 40 das 40 colunas mudam por fold. | Etapa 4 0,4519 vs. Etapa 5 0,4382 ± 0,0133 | Alta |
| 5 | **48 de 136** colunas pré-selecionadas têm importância por permutação negativa — prejudicam ativamente. | `feature_selection_ranking_deployed.csv` | Alta |
| 6 | A concentração de importância não prevê a suficiência: as 10 primeiras colunas concentram 84% da importância, mas K = 20 falha por 1,6 pontos. | Varrimento de K | Alta |

**A resposta numa linha.** 40 colunas preservam a pontuação validada cruzada da
linha de base dentro do ruído, mas um otimismo de seleção de +0,0137 coloca a
estimativa honesta em **~0,438**, que é **inferior** à linha de base de 255
colunas de 0,4531. Se essa diferença é real é exatamente o que as duas pontuações
do Kaggle irão decidir, porque o conjunto de teste nunca viu a seleção. **Essas
pontuações ainda não existem; este documento está marcado como pendente nesse
ponto.**

---

## 3. Visita guiada: a matéria que o curso não cobre

O curso cobre a justificação da seleção de variáveis e os métodos wrapper/embedded
(Aula2b 20–22), codificação por rótulos vs. one-hot (Aula2b 24), importância por
permutação e MDI (PL_7 26–33), validação cruzada k-fold (PL_2 39–48), e
precisão/revocação/F1 (PL_2 14–27). Tudo o que se segue **não** está nesses
diapositivos. Cada entrada diz o que é, porque esta etapa precisou dele, e a
armadilha que ele prepara.

### 3.1 F1 macro

**O que é.** O F1 simples (PL_2 diapositivo 26) está definido para duas classes.
Com 16 classes é preciso escolher como combinar 16 pontuações F1 por classe. O F1
**macro** é a média não ponderada das 16 pontuações por classe. O F1 **ponderado**
pesa cada classe pelo seu suporte. O F1 **micro** é simplesmente a exatidão
(accuracy) em problemas de rótulo único.

**Porque importa aqui.** A competição avalia com F1 macro, por isso todos os
números deste documento são F1 macro. A diferença não é cosmética: a faixa 0 tem
207 linhas (1,9%), a faixa 10 tem 1.062. Com média macro, a faixa de 207 linhas
conta tanto como a de 1.062, pelo que um modelo bom nas faixas grandes e mau nas
raras obtém uma pontuação muito pior em macro do que em ponderado ou micro.

**A armadilha.** É por isso que a comparação arquivada de `data_v1` não é
comparável com esta. Esse arquivo mediu **exatidão** e reportou que o top-20 perdia
4,3 pontos percentuais. O F1 macro dá mais peso às faixas raras, por isso o mesmo
corte de variáveis custa mais sob F1 macro. Nunca colocar o valor de 4,3 pontos ao
lado de um número de F1 macro.

### 3.2 Validação cruzada repetida

**O que é.** O curso ensina uma passagem de k-fold: dividir os dados em k folds,
treinar k vezes, fazer a média das k pontuações. A validação cruzada **repetida**
faz todo esse procedimento R vezes com partições aleatórias diferentes. Aqui
R = 3 e k = 5, o que dá **15 folds**.

**Porque importa aqui.** Uma única passagem de 5 folds dá 5 números. O seu desvio
padrão é uma estimativa de quanto a pontuação varia com a *divisão*, e com 5
números essa estimativa é ela própria muito ruidosa. Repetir dá 15 números e uma
estimativa mais estável. Todos os "±" deste documento são o desvio padrão ao longo
desses 15 folds.

**A armadilha.** Os 15 folds **não são independentes**. As três repetições avaliam
as mesmas linhas, apenas particionadas de forma diferente. Por isso o desvio
padrão entre folds é uma dispersão **descritiva**, não um intervalo de confiança.
Não calcular "média ± 2 EP" a partir dele e não o tratar como um teste de
significância. Isto está dito no notebook precisamente por esta razão.

### 3.3 Comparação emparelhada em folds idênticos

**O que é.** Ao comparar dois modelos, compará-los **fold a fold nas mesmas
linhas**, e não comparando duas médias. Para o fold *i*, calcular
`baseline_i − candidate_i`; depois olhar para a distribuição dessas diferenças.

**Porque importa aqui.** Dois modelos avaliados no mesmo fold partilham a
dificuldade desse fold. O fold 2 é difícil para ambos; o fold 4 é fácil para
ambos. A diferenciação cancela essa componente partilhada, pelo que a diferença
emparelhada é muito menos ruidosa do que as duas pontuações em bruto. A coluna
`paired_difference_mean` do varrimento de K é exatamente isto: linha de base menos
K, por fold, e depois a média.

**A armadilha.** Uma diferença emparelhada próxima de zero com grande dispersão
*não* é evidência de igualdade. É evidência de **nenhuma diferença detetável com
esta dimensão de amostra**. A distinção importa para a constatação 1: não se
demonstrou que 40 colunas são tão boas como 255, apenas que não se demonstrou que
são piores.

### 3.4 Pré-registo de uma regra de decisão

**O que é.** Escrever a regra de decisão **antes** de ver o resultado e depois
aplicá-la mecanicamente. É prática padrão em investigação e não é um tópico do
curso.

**Porque importa aqui.** O modo de falha tentador é correr o varrimento, ver que
K = 60 tem a melhor pontuação, e depois definir "dentro do ruído" de forma a que
60 passe. Isso seria escolher a regra para se ajustar à resposta. Duas regras
nesta etapa foram fixadas antecipadamente e são citadas literalmente na secção 4.

**A armadilha.** O pré-registo não torna a regra *correta*; torna-a **honesta**.
Uma regra mal escolhida aplicada mecanicamente continua a ser uma regra mal
escolhida. A constatação 2 é exatamente este caso: a regra foi seguida, e a regra
é discutivelmente demasiado rígida. Ambas as afirmações pertencem ao registo.

### 3.5 Fuga de seleção e seleção aninhada

**O que é.** Esta é a ideia mais importante da etapa e o curso não a enuncia. O
diapositivo 20 da Aula2b diz que a seleção de variáveis "reduz o sobreajuste" e
fica por aí. Nunca diz que **o ato de selecionar variáveis usando rótulos que
depois aparecem num fold de validação inflaciona a pontuação que se mede**.

O mecanismo: as Etapas 1 e 2 olham para os rótulos de **todas as linhas** para
decidir que colunas são importantes. Com validação cruzada de 5 folds, cada linha
está em algum fold de validação. Logo, o rótulo de cada linha influenciou a
escolha das colunas, e depois essas mesmas linhas foram usadas para avaliar o
modelo. A pontuação é otimista.

**Porque importa aqui.** Os folds congelados protegem o *pré-processador* de
fugas — o escalonador e o codificador são ajustados dentro de cada fold. **Não**
protegem contra olhar para pontuações de validação enquanto se escolhem colunas.
Essa é uma fuga diferente, e é invisível a menos que se procure por ela.

**A correção — seleção aninhada.** Correr o procedimento de seleção *completo*
dentro de cada fold externo:

1. Tomar apenas as linhas de treino do fold externo.
2. Retirar delas uma amostra interna de retenção de 20%.
3. Correr a pré-seleção por informação mútua nos restantes 80%, ajustar o
   ordenador aí, e ordenar por permutação **na amostra de retenção** — de modo a
   que a permutação seja fora de amostra, tal como na Etapa 2.
4. Tomar as K primeiras, ajustar o modelo final nas linhas de treino externas,
   avaliar no fold de validação externo.

**A armadilha.** Se ajustar o ordenador a todas as linhas de treino externas e
depois permutar num subconjunto delas, a permutação é **em amostra**. Isso
inflaciona colunas de alta cardinalidade e contamina a diferença que se está a
tentar medir. O notebook usa uma amostra de retenção genuína por esta razão.

### 3.6 Informação mútua com correção de enviesamento por baralhamento

**O que é.** A informação mútua (IM) mede quanto é que conhecer uma coluna reduz a
incerteza sobre o rótulo. Deteta dependência não linear, ao contrário da
correlação. O curso oferece o teste de **qui-quadrado** para dependência
variável/alvo (Aula2b diapositivo 21); a IM é uma ferramenta diferente e mais
geral.

**Porque importa aqui.** A Etapa 1 usa a IM como primeira pré-seleção barata,
reduzindo 255 entradas a 136 colunas candidatas antes da ordenação por permutação,
que é cara.

**A armadilha — a IM é enviesada para cima pela cardinalidade.** Uma coluna com
muitos valores distintos pode parecer informativa só por acaso. A correção usada
aqui é **baralhar os rótulos**, recalcular a IM e subtrair o valor baralhado. O que
sobrevive é sinal acima do que a associação aleatória produz. Sem esta correção, a
IM favorece sistematicamente colunas de alta cardinalidade.

### 3.7 Importância por permutação em F1 macro

**O que é.** Os diapositivos 30–31 de PL_7 introduzem a importância por
permutação: baralhar uma coluna e medir quanto a pontuação cai. O curso demonstra-a
com **exatidão**. Aqui o avaliador é o **F1 macro**.

**Porque importa aqui.** A importância de uma coluna é definida *em relação à
métrica*. Baralhar uma coluna que só importa para as faixas raras quase não mexe na
exatidão mas mexe muito no F1 macro. Assim, esta ordenação responde a "que colunas
importam para o F1 macro", que é a métrica da competição.

**A armadilha — valores negativos têm significado.** Uma coluna com importância
negativa *melhorou* a pontuação quando foi baralhada, ou seja, estava a prejudicar
ativamente nos folds de validação. Isto não é um erro de programação. **48 das 136
colunas pré-selecionadas têm importância negativa** (constatação 5). São evidência
real de que essas colunas transportam ruído que o modelo estava a ajustar.

### 3.8 Correlação de ordem de Spearman

**O que é.** Uma correlação entre duas **ordenações**, não entre dois valores.
Pergunta: quando a ordenação A coloca uma coluna no topo, a ordenação B também a
coloca no topo? Varia de −1 a +1 e ignora as magnitudes, apenas a ordem.

**Porque importa aqui.** A Etapa 3 compara duas ordenações das mesmas 20 colunas
produzidas por duas pipelines diferentes. A correlação de Pearson estaria errada —
os valores de importância têm escalas completamente diferentes (0,005 vs. 0,174).
Spearman só se interessa pela ordem, que é o que significa "estas duas ordenações
concordam?".

**A armadilha.** Um Spearman de 0,492 diz que as ordenações **discordam**. Não diz
qual delas está certa. Não existe verdade absoluta para "a importância correta"; a
única verificação externa é a pontuação da competição.

### 3.9 Enviesamento da codificação ordinal nas importâncias de árvores

**O que é.** PL_7 diz que a MDI é enviesada para variáveis de alta cardinalidade.
Este é um efeito **diferente e mais acentuado**: uma coluna **nominal** (sem ordem
natural) codificada por rótulos fica com uma ordem numérica arbitrária. Uma árvore
tem então de fazer vários cortes para isolar uma categoria que a codificação
one-hot isolaria num só.

**Porque importa aqui.** Foi o que `PC1A` demonstrou. Com códigos ordinais, a
floresta aleatória colocava-a em **33.º** com importância 0,005. Com a pipeline
one-hot implementada, fica em **1.º** com importância 0,174 — um fator de 35. Uma
nominal de 3 níveis precisa de vários cortes com códigos ordinais e de um só com
one-hot, por isso a lente ordinal subestima-a sistematicamente.

**A armadilha.** A Etapa 3 muda **duas coisas ao mesmo tempo** — o modelo
(RF → `hist_gb`) e a representação (ordinal → one-hot) — pelo que o 0,492 não pode
ser atribuído apenas à codificação. Com n = 20 as barras de erro são largas. A
conclusão é qualitativa: a escolha de representação mudou a resposta.

### 3.10 `HistGradientBoostingClassifier` e `passthrough`

**O que é.** O curso cobre boosting e XGBoost (Aula5 diapositivos 8–10). Esta
etapa usa o `HistGradientBoostingClassifier` do scikit-learn, que agrupa variáveis
contínuas em histogramas por razões de velocidade e trata valores em falta de forma
nativa. `passthrough` significa que o pré-processador deixa passar as colunas em
bruto sem alteração, deixando o modelo fazer os seus próprios cortes em vez de
receber colunas já codificadas.

**Porque importa aqui.** É o modelo que o grupo implementa de facto, por isso a
ordenação tem de ser produzida por *este* modelo com *esta* representação. Ordenar
com qualquer outra configuração mede essa configuração, não a implementada.

**A armadilha.** O tratamento nativo de categóricas significa que o modelo pode ver
uma coluna nominal como uma categoria, não como um número — que é precisamente a
razão pela qual discorda da RF ordinal. A discordância é uma propriedade da
pipeline, não um erro em nenhuma das ordenações.

### 3.11 Impressões digitais e proveniência

**O que é.** Uma impressão digital (fingerprint) é um hash de um conjunto de nomes
de colunas (e dos valores do alvo). A proveniência é o registo do que produziu um
artefacto: o SHA do git, se a árvore estava suja, a semente, a especificação dos
folds.

**Porque importa aqui.** O `selected_features.json` guarda ambos. O carregador
verifica novamente a impressão digital quando o ficheiro é lido: se a lista final
já não corresponde aos dados, **falha de forma ruidosa** em vez de descartar uma
coluna em silêncio. Uma lista final que deixou discretamente de se ajustar aos
dados é pior do que não ter lista nenhuma.

**A armadilha.** Uma impressão digital protege contra *desvio*, não contra *erro*.
Confirma que as colunas são as que foram escolhidas; não diz nada sobre se
escolhê-las foi uma boa ideia.

---

## 4. Método tal como executado

### 4.1 Etapas

| Etapa | O que faz | Custo |
|---|---|---|
| 0 | Linha de base: `hist_gb` com as 255 entradas, 15 folds | ~5 min |
| 1 | Pré-seleção por informação mútua com correção por baralhamento: 255 → 136 colunas | rápida |
| 2 | Ordenação por permutação das 136, RF com codificação ordinal, 5 repetições | ~28 min |
| 3 | Verificação de consistência da representação: reordenar o top 20 com a pipeline implementada | ~5 min |
| 3b | **Alternativa, correu aqui:** reordenar as 136 com `hist_gb` + `passthrough` | ~30 min |
| 4 | Varrimento de K sobre {10, 15, 20, 30, 40, 60, 80}, um modelo, emparelhado contra a linha de base | ~20 min |
| 5 | Seleção aninhada: todo o procedimento refeito dentro de cada fold externo | ~22 min |
| 6 | Persistência com impressões digitais | rápida |
| 7 | Duas submissões: selecionadas K=40 e todas as variáveis | ~15 min |

O tempo total de parede foi de cerca de 2 h 20 min, incluindo a alternativa da
Etapa 3.

### 4.2 As duas regras pré-registadas, à letra

**Ramo da Etapa 3** (fixado antes da execução; citado do notebook):

> Se a correlação for alta (> 0,7), a lente ordinal serve e a lista final é
> construída a partir da ordenação da Etapa 2. Se for baixa, a lista final é
> construída a partir da representação que o modelo usa de facto: todo o conjunto
> pré-selecionado é reordenado com `hist_gb` + `passthrough`.
>
> Este é um ramo pré-registado, não uma escolha feita depois de ver a pontuação. A
> única coisa que o número decide é *qual ordenação já declarada vence*.

**Regra de aprovação da Etapa 4** (fixada antes da execução; citada do notebook):

> K passa se a diferença emparelhada média ao longo dos folds for **no máximo
> 0,005** e nenhuma repetição isolada mostrar uma perda pior do que 0,010.

**Alternativa pré-registada:** se nenhum K passasse, nem mesmo K = 80, a lista
final seria o conjunto completo de entradas e o relatório diria que a seleção não
sobreviveu à validação. Isto não foi acionado.

### 4.3 Porque a Etapa 5 espelha o vencedor da Etapa 3

A Etapa 5 usa o **mesmo ordenador que produziu a lista final**: a pipeline
`hist_gb` implementada, porque a Etapa 3 falhou. Se tivesse usado a RF ordinal, a
diferença mediria a *discordância entre dois ordenadores* em vez do *otimismo de
seleção*. Espelhar mantém a medição limpa.

---

## 5. Resultados

### 5.1 Linha de base e o varrimento de K

Linha de base: `hist_gb` com as 255 entradas, **0,4531 ± 0,0096**
(execução `20261103T225840834739_hist-gb-all-recoded`).

A diferença emparelhada é **linha de base menos K**, por isso um número positivo é
uma perda.

| K | F1 macro | desvio padrão | diferença emparelhada | pior repetição | passa |
|---:|---:|---:|---:|---:|:--:|
| 10 | 0,4139 | 0,0096 | +0,0392 | +0,0347 | não |
| 15 | 0,4294 | 0,0093 | +0,0237 | +0,0199 | não |
| 20 | 0,4367 | 0,0094 | +0,0164 | +0,0100 | não |
| 30 | 0,4403 | 0,0104 | +0,0128 | +0,0098 | não |
| **40** | **0,4519** | **0,0104** | **+0,0012** | **−0,0049** | **sim** |
| 60 | **0,4587** | 0,0072 | −0,0056 | **−0,0102** | não |
| 80 | 0,4574 | 0,0090 | −0,0044 | −0,0070 | sim |

**K = 40 é o menor conjunto que passa a regra.**

### 5.2 A verificação da Etapa 3

Correlação de Spearman entre as ordenações da RF ordinal e da pipeline
implementada: **0,492**. Abaixo de 0,7, por isso a alternativa correu e a lista
final vem da ordenação implementada
(`ranking_source: stage3_deployed_hist_gb`).

A discordância mais consequente:

| coluna | posição ordinal | posição implementada | importância ordinal | importância implementada |
|---|---:|---:|---:|---:|
| `PC1A` | 33 | **1** | 0,005 | **0,174** |

### 5.3 A verificação aninhada

| fold | K | F1 macro | colunas diferentes da Etapa 4 |
|---:|---:|---:|---:|
| 0 | 40 | 0,4498 | 26 |
| 1 | 40 | 0,4278 | 40 |
| 2 | 40 | 0,4320 | 34 |
| 3 | 40 | 0,4551 | 38 |
| 4 | 40 | 0,4262 | 38 |

Média da Etapa 5: **0,4382 ± 0,0133**. Etapa 4 para o mesmo K: 0,4519.
**Otimismo de seleção: +0,0137.**

### 5.4 Comportamento por classe, K=40 contra a linha de base

Ver §6.1 para a interpretação desta tabela.

| faixa | F1 linha de base | F1 K=40 | delta | suporte |
|---:|---:|---:|---:|---:|
| 0 | 0,857 | 0,829 | −0,028 | 207 |
| 1 | 0,308 | **0,350** | **+0,041** | 172 |
| 2 | 0,601 | 0,585 | −0,016 | 399 |
| 3 | 0,373 | **0,388** | **+0,015** | 377 |
| 4 | 0,259 | **0,277** | **+0,018** | 449 |
| 5 | 0,347 | 0,344 | −0,003 | 647 |
| 6 | 0,358 | 0,340 | −0,018 | 859 |
| 7 | 0,389 | **0,401** | **+0,013** | 862 |
| 8 | 0,634 | 0,621 | −0,013 | 905 |
| 9 | 0,661 | 0,635 | −0,026 | 1014 |
| 10 | 0,645 | 0,619 | −0,026 | 1062 |
| 11 | 0,514 | **0,516** | **+0,002** | 1040 |
| 12 | 0,380 | 0,353 | −0,027 | 1000 |
| 13 | 0,280 | 0,252 | −0,028 | 823 |
| 14 | 0,313 | **0,314** | **+0,001** | 650 |
| 15 | 0,449 | 0,387 | **−0,063** | 496 |

**6 classes melhoram, 10 pioram, delta médio −0,0098.**

### 5.5 A lista final

40 colunas, pela ordem de importância implementada:

`PC1A`, `MAINSTAT`, `HH_INACT`, `MARSTALEGAL`, `HATLEVEL`, `PARTNERS`,
`HH_ACT`, `PA7_NAC`, `HHTYPE`, `HHNBPERS_0_13_COD`, `IN45`, `HHNBPERS_COD`,
`JOBSTAT_COD`, `NT1`, `IN47`, `BM2`, `PA1`, `IN63`, `HA2`, `HHINCOME`,
`FT_PT`, `BM1`, `IN65`, `PA5`, `PE9`, `PL9_tot`, `IN4`, `IN24`, `IN28`,
`IN15`, `MD1`, `SK2`, `IN16`, `PC2`, `DH6`, `HA1G`, `JOBISCO_COD`, `PL5`,
`AM3`, `IN22`

Por tema:

- **Situação perante o trabalho e emprego** — `MAINSTAT`, `JOBSTAT_COD`,
  `JOBISCO_COD`, `FT_PT`, `HH_ACT`, `HH_INACT`
- **Educação** — `HATLEVEL`, `PE9`, `NT1`
- **Estrutura do agregado familiar** — `HHTYPE`, `HHNBPERS_COD`,
  `HHNBPERS_0_13_COD`, `PARTNERS`, `MARSTALEGAL`
- **Rendimento e riqueza** — `HHINCOME`, `PL9_tot`, `PL5`, `MD1`
- **Saúde e incapacidade** — `HA2`, `HA1G`, `DH6`
- **Migração e nacionalidade** — `PA7_NAC`, `PA1`, `PA5`, `BM1`, `BM2`
- **Região** — `PC1A`, `PC2`
- **Atitudes e opiniões** — 10 das 40 são colunas `IN*`

**Blocos nunca selecionados:** `CD*` (9 colunas), `AL*` (4), `HS*` (3), `AW*` (2),
`HO*` (2), `PN*` (2), `UN*` (2), `LOCNACE` (1). Não transportam sinal de
permutação para este alvo.

---

## 6. Interpretação

### 6.1 Quarenta colunas igualam a linha de base em termos agregados, mas reorganizam por classe

A evidência é a tabela do §5.4.

A diferença agregada é +0,0012, cerca de um oitavo de um desvio padrão. A leitura
honesta é **"nenhuma perda detetável"**, não "40 colunas são tão boas como 255".

Mas a tabela por classe mostra que este não é um corte neutro. Seis classes
melhoram — sobretudo as faixas intermédias esparsas (1, 3, 4, 7) — e dez pioram,
incluindo as faixas bem povoadas 9, 10 e 12 e, pior de tudo, **a faixa 15 com
−0,063**. O modelo de 40 colunas não é o modelo de 255 colunas com menos ruído; é
um modelo **diferente** que troca algumas classes por outras e por acaso fica na
mesma média.

Isso importa para o objetivo declarado do projeto. O enunciado pede que as
previsões do modelo sejam usadas para detetar anomalias de perfil, o que exige que
as faixas intermédias estejam bem separadas. Um corte que melhora as faixas 1 e 4
ao mesmo tempo que perde a faixa 15 não é obviamente a troca certa para esse fim,
mesmo que o F1 macro não consiga distinguir a diferença.

### 6.2 K = 60 tem a melhor média e a regra rejeitou-o

K = 60 obtém **0,4587**, melhor do que a linha de base na média e melhor do que
K = 40. Falha a guarda por **0,0002** (−0,0102 contra um limiar de −0,010).

Duas coisas são verdadeiras ao mesmo tempo e ambas pertencem ao registo:

- **A regra foi fixada antecipadamente e foi seguida.** É esse o objetivo do
  pré-registo, e a guarda fez exatamente aquilo para que foi concebida.
- **A guarda está no fio da navalha com esta dimensão de amostra.** Uma diferença
  de 0,0002 é ruído, não evidência de instabilidade. K = 60 é *plausivelmente* o
  melhor ponto de operação, e foi a regra — não os dados — que escolheu 40.

O varrimento mostra também **K = 80 a passar** (0,4574, diferença −0,0044)
enquanto K = 60 falha. Essa não monotonicidade é um aviso: a guarda está em parte a
medir ruído entre folds, e não apenas a qualidade do modelo. Uma regra que rejeita
60 mas aceita 80 não está a seguir um limiar real.

**Esta é uma decisão para os humanos, não um resultado fechado.** As opções estão
na secção 8.

### 6.3 A verificação da Etapa 3 compensou o seu custo

A RF ordinal colocava `PC1A` em 33.º, com importância 0,0021; a pipeline
implementada coloca-a em 1.º, com importância 0,1741 — um **aumento de 82×**. Se a
verificação tivesse passado, `PC1A` teria sido **retirada da lista final**, e é a
coluna isolada mais importante sob o modelo que o grupo entrega.

A lição metodológica é a parte reutilizável: **ordenar as variáveis com a
representação que o modelo usa, ou a ordenação mede a codificação e não os dados.**
A lente ordinal não estava errada de uma forma que se manifestasse como erro;
estava errada de uma forma que reordenou silenciosamente a resposta.

Ressalva: a verificação muda o modelo e a representação em conjunto, por isso o
0,492 não pode ser atribuído apenas à codificação, e n = 20 é uma barra de erro
larga.

### 6.4 O otimismo de seleção é +0,0137, e a lista final é instável

A Etapa 4 diz 0,4519. A Etapa 5 — o mesmo procedimento com a seleção refeita
dentro de cada fold — diz **0,4382**. Cerca de **1,4 pontos da pontuação aparente
foram comprados ao olhar para os rótulos**.

As contagens de colunas por fold são a constatação mais afiada: **26 a 40 das 40
colunas mudam em cada fold, e num fold mudam as 40.** A lista final exata não é um
objeto estável. O que a Etapa 5 mede é o **procedimento**, não a lista.

**A consequência desconfortável:** a estimativa honesta de ~0,438 é **inferior** à
linha de base de 255 colunas de 0,4531. Com esta evidência, a afirmação defensável
não é "40 colunas igualam a linha de base" mas:

> 40 colunas igualam a pontuação validada cruzada da linha de base, mas, uma vez
> contabilizado o custo de as escolher, o procedimento pode ser ligeiramente pior
> do que usar tudo.

Se essa diferença é real está por resolver. A estimativa validada cruzada é
otimista para o conjunto selecionado e não para a linha de base, por isso os dois
números não são diretamente comparáveis — mas o conjunto de teste nunca viu a
seleção, por isso as duas pontuações do Kaggle são a comparação limpa. **Estão
pendentes.**

### 6.5 Quarenta e oito colunas prejudicam ativamente

48 das 136 colunas pré-selecionadas têm importância por permutação **negativa**:
baralhá-las *melhorou* o F1 macro nos folds de validação. Isto é mais forte do que
"as restantes não são importantes" — diz que o modelo estava a ajustar ruído
através dessas colunas.

É evidência direta para a afirmação do próprio curso (Aula2b diapositivo 20) de
que remover variáveis pouco importantes reduz o sobreajuste. É também uma pista:
um modelo treinado nas 88 colunas não negativas pode superar tanto as 40 como as
255.

### 6.6 A concentração de importância não prevê a suficiência

As 10 primeiras colunas concentram **84%** da importância total, e as 20 primeiras
concentram **95%** — no entanto K = 20 falha a regra por 1,6 pontos. A concentração
de importância e a suficiência do modelo são questões diferentes. Uma coluna pode
importar muito e continuar a ser redundante com outra: a importância por
permutação reparte o crédito entre colunas correlacionadas, por isso uma coluna
pode parecer fraca enquanto o seu grupo é essencial.

É o mesmo aviso que o notebook registra sobre `HA1*`/`PC1*`: o seu nível simples
"não aplicável" é 100% para menores de 55 anos e 0% para 55 ou mais, por isso um
único corte aos 55 capta todas. São **um único sinal de elegibilidade por idade,
não treze achados de saúde**. Reportar temas, não treze descobertas.

---

## 7. Porque não foram usados os métodos do próprio curso

Esta secção existe porque o curso oferece várias técnicas de seleção de variáveis e
esta etapa usou outras. Cada afastamento é deliberado, e um leitor que só conhece a
matéria do curso tem o direito de perguntar porquê.

A taxonomia do diapositivo 22 da Aula2b é o enquadramento: os **filtros** avaliam
cada variável contra o alvo sem modelo (qui-quadrado); os **wrappers** usam um
modelo e medem quanto o desempenho cai sem a variável (SFS, SBE); os métodos
**embedded** selecionam à medida que treinam (LASSO, impureza de árvores). Esta
etapa é **filtro e depois wrapper** — a informação mútua pré-seleciona 255 → 136, a
importância por permutação ordena essas 136.

| Método do curso | Onde | Porque não foi usado | Veredicto |
|---|---|---|---|
| **Teste de qui-quadrado** | Aula2b 21 | Univariado e só para categóricas. Exigiria discretizar as 31 colunas numéricas, o que perde informação. A IM trata tipos mistos de forma uniforme e acrescenta uma correção de enviesamento por baralhamento. | **Alternativa defensável.** O qui-quadrado teria sido uma escolha razoável; a IM foi preferida, não exigida. |
| **Remover baixo desvio padrão / assimétricas / altamente correlacionadas** | Aula2b 21 | São regras de higiene, não seleção sensível ao alvo. Respondem a "esta coluna comporta-se bem?", não a "prevê a idade?". | Aplicadas implicitamente pela recodificação; não usadas como critério de seleção. |
| **Wrapper SFS / SBE** | Aula2b 22 | A pesquisa combinatória sobre 136 colunas × 15 folds é inviável. | **Substituído, não omitido.** A importância por permutação é o wrapper tratável: mede a utilidade pelo desempenho do classificador, que é a própria definição do diapositivo. |
| **Embedded / LASSO** | Aula2b 22 | L1 sobre 1.170 colunas one-hot com 16 classes é instável, e a seleção seria linear. | Não usado. Precisaria da sua própria validação. |
| **PCA** | Aula2b 22 | Destrói a interpretabilidade, que o enunciado exige explicitamente ("analisar os fatores que mais contribuem"). | Rejeitado por razões de requisito, não de desempenho. |
| **Importância por permutação** | PL_7 27–31 | — | **Usado.** É o núcleo do método. |
| **Impureza de árvores (MDI)** | PL_7 30 | Enviesada para colunas de alta cardinalidade; em amostra. | Usada apenas como ordenação interna da RF na Etapa 2, depois substituída. |

**O resumo honesto.** O núcleo — ordenar variáveis pela importância por permutação
do F1 macro de um modelo — é PL_7 mais o diapositivo 22 da Aula2b. As extensões são
a disciplina de validação em torno dele: validação cruzada repetida, comparação
emparelhada, uma regra pré-registada e seleção aninhada.

O único afastamento genuíno é a **seleção aninhada**, e existe para *testar* a
afirmação da aula (Aula2b diapositivo 20) de que remover variáveis pouco
importantes reduz o sobreajuste. O resultado é que a afirmação é **condicional**: o
ganho aparente é em parte comprado ao olhar para os rótulos. É um contributo para o
argumento do curso, não um desvio dele. A secção 6.5 é a outra metade do mesmo
teste — 48 colunas com importância negativa são evidência direta *a favor* do
diapositivo.

---

## 8. Decisões para os humanos

Estas estão em aberto. Esta etapa informa-as; não as resolve.

**D1. Que K entregar: 40 ou 60?**
A regra escolheu 40. A média favorece 60. Um caminho intermédio defensável é
submeter ambos ao Kaggle e deixar o conjunto de teste decidir, o que custa duas das
três submissões diárias. **Recomendação:** submeter K=40 e o modelo com todas as
variáveis primeiro (ambos já construídos), depois K=60 se o orçamento diário o
permitir.

**D2. Vale a pena manter a seleção de todo?**
Se a pontuação do Kaggle para K=40 não for pelo menos tão boa como a de todas as
variáveis, a conclusão honesta é que a seleção não sobreviveu à validação, e o
grupo deve entregar o modelo completo. Isto era a alternativa pré-registada para a
*regra*; agora é uma questão em aberto para o *conjunto de teste*.

**D3. Deve a guarda ser alterada?**
A guarda de −0,010 por repetição rejeitou 60 e aceitou 80. Se o grupo quiser
continuar a usar uma guarda, deve derivá-la de novo — por exemplo a partir da
dispersão observada entre repetições em vez de um número redondo. Alterá-la agora
seria post-hoc, por isso qualquer alteração tem de ser declarada para a *próxima*
etapa, não aplicada retroativamente.

**D4. O que fazer com as 48 colunas de importância negativa?**
Uma execução nas 88 colunas não negativas é barata e é uma hipótese genuína, não
uma expedição à procura de resultados, porque o sinal foi medido fora de amostra
nos folds de validação.

**D5. As colunas de atitude `IN*`.**
10 das 40 colunas selecionadas são atitudes e opiniões. Ficam no topo com
`hist_gb` e no fundo com a RF ordinal. Ou é sinal real, ou é o mesmo artefacto de
representação. **Não construir uma narrativa sobre elas** até alguém verificar o
que essas colunas significam e se o efeito sobrevive a uma representação diferente.

---

## 9. Limitações

1. **As pontuações do Kaggle estão pendentes.** A afirmação central da secção 6.4
   fica por resolver sem elas. Tudo o resto é apenas evidência validada cruzada.
2. **A verificação aninhada usa 5 folds externos e uma divisão interna por fold.**
   A estimativa de otimismo é ela própria ruidosa; ±0,0133 entre folds.
3. **Os 15 folds não são independentes**, por isso não é calculado nenhum teste de
   significância e nenhum deve ser.
4. **A verificação da Etapa 3 está confundida** por construção (o modelo e a
   representação mudam em conjunto) e usa n = 20.
5. **O mapeamento da faixa 0 está em disputa** (`TODO.md`). Nenhum resultado aqui
   depende dele, porque os rótulos são tratados como códigos, mas nenhuma
   interpretação de idade deve ser publicada sem esta ressalva.
6. **As saídas guardadas de `reports/01` são de `data_v1`** e mediram exatidão, não
   F1 macro. Não são comparáveis e não são usadas aqui.
7. **A importância por permutação reparte o crédito entre colunas
   correlacionadas.** Um único membro de um grupo correlacionado pode parecer fraco
   enquanto o grupo é essencial. É por isso que a secção 6.6 reporta temas.
8. **Nada aqui é causal.** `HATLEVEL` estar no topo não significa que a educação
   cause a faixa etária; ambos refletem a fase da vida.

---

## 10. Reprodução

```bash
# A partir da raiz do repositório, em c60f8f8 com a árvore limpa:
.venv/bin/jupyter nbconvert --to notebook --execute \
  notebooks/04_feature_selection.ipynb \
  --output-dir /tmp/opencode --output 04_executed.ipynb \
  --ExecutePreprocessor.timeout=14400
```

O notebook versionado corre por omissão em modo de simulação (`SMOKE = True`,
~14 min, números sem significado). Para a execução real, definir `SMOKE = False`
numa cópia e executar essa cópia; os artefactos ficam em `outputs/`
independentemente, porque os caminhos vêm de `daa.config`.

Artefactos escritos:

| ficheiro | conteúdo |
|---|---|
| `outputs/selected_features.json` | a lista final de 40 colunas, impressões digitais, proveniência |
| `outputs/feature_selection_k_sweep.csv` | K, F1 macro, diferença emparelhada, indicador de aprovação, id da execução |
| `outputs/feature_selection_ranking.csv` | ordenação da Etapa 2 pela RF ordinal, todas as 136 |
| `outputs/feature_selection_ranking_deployed.csv` | ordenação implementada com `hist_gb`, todas as 136 |
| `outputs/submission_selected_k40.csv` | submissão, K=40 |
| `outputs/submission_all_features.csv` | submissão, todas as 255 entradas |

Ambas as submissões validadas: 3.655 linhas, `RowId` 0…3654, 16 classes.

**Pendente:** carregar ambas para o Kaggle e registar as duas pontuações de F1
macro. É a única evidência que falta para fechar a secção 6.4.
