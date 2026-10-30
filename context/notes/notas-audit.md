<!--
Audit trail for the extraction from notas.md. Every original line is either
represented in ml-reference.md, listed as a drop, or listed as a standard-reference
addition. Line numbers refer to the backup copy
(context.bak-2026-10-30/notas.md).

Status: 13 corrections, 16 drops, 8 standard-reference additions.
The line-range map is the losslessness check.
-->

# Audit trail — `notas.md` → `ml-reference.md`

## Line-range map

| Original lines | Destination |
|---|---|
| 1–7 | §1 |
| 9–11 | §2 |
| 13–23 | §2 |
| 25–31 | §3 (L29 and L31 handled: L29 corrected, L31 dropped) |
| 33–39 | §2 |
| 41–54 | §4 |
| 56–62 | §4 |
| 64–66 | §7 (histogram as a labelled rule of thumb) |
| 68–70 | dropped (tooling opinion) |
| 72–86 | §5 |
| 88–106 | §7 (L94 noise definition retained in the missing/noise checklist) |
| 108–116 | §5 and §6 (L110 prediction dropped; L112–114 anecdote dropped; L116 concept drift) |
| 118–139 | §6 (L128, L134 dropped as opinions; L130 survivorship; L132 baseline; L136 generalisation; L138–139 rule of thumb) |
| 141–182 | §7 (L144–145 reworded; L147–149 outliers; L151–157 selection; L159–164 missing; L166–170 encoding; L172 heading covered by §7; L174 split order; L176–178 corrected; L180 pair plot; L182 dropped) |
| 184–207 | §8 (L188 corrected; L193 dropped; L197 corrected; L199–203 estimation; L205–207 regularisation) |
| 209–237 | §9 (L223 "heavy" dropped as vague; L229–231 stopping including L230–231; L235–237 pruning) |
| 239–240 | §6 (cross-validation spread) |
| 242–259 | §10 (L246 "série/paralelo/?" reworded; L248–251 corrected; L252–253 stacking; L255; L257; L259 `n_estimators`) |
| 261–278 | §11 (L266 activation corrected; L270 "world" dropped as opinion; L272 vanilla retained; L274–276; L278) |
| 280 | empty |

## Corrections applied

| Original line | Was | Now | Status |
|---|---|---|---|
| L4–6 | nominal/ordinal "erros binários/discretos" | error cost follows the type; ordinal **can** use a rank distance | corrected |
| L45 | birth date ↔ age "covariância direta" | negatively correlated | corrected |
| L47 | "covariância é a média das variâncias" | mean of co-deviations | corrected |
| L86 | RL "não supervisionado com características de supervisionado" | third paradigm, reward signal | corrected |
| L144–145 | min-max "preferível sem viés", standardisation "com viés" | standardisation is a **rule of thumb** for skewed/outlier-heavy data | reworded as heuristic |
| L177 | "kurtosis negativo: montanha esguia" | negative = flat, light tails | corrected |
| L178 | "skewness positivo: assimetria à esquerda" | positive = right tail | corrected |
| L188 | `f(x+y) = f(x)·f(y)` | `f(x+y) = f(x)+f(y)` | corrected |
| L197 | "multiplico as classes…" | multinomial (softmax) / one-vs-rest | corrected |
| L248 | bagging on "subsets (ex 70%)" | bootstrap, n draws with replacement | corrected |
| L251 | "Xg altera os hiperparâmetros" | fits gradients + regularisation; user tunes hyperparameters | corrected |
| L263 | NN "ignoram ruído e inconsistências" | they can overfit noise | corrected |
| L266 | "regressão usa linear ou ReLU para valores não negativos" | linear for regression; ReLU for non-negative outputs | corrected |

## Drops

| Original line | Text | Why |
|---|---|---|
| L17 | "A variância quer se grande, para podermos extrair informação" | not a meaningful statement |
| L31 | "Também é preciso fazer para variáveis categóricas" | garbled, no referent |
| L66 | "O histograma tem de aparecer sempre" | opinion → only the labelled first-plot heuristic is kept |
| L70 | "Nao ao Anaconda sim ao miniconda" | tooling opinion; the repo uses `venv` |
| L110 | "O prof acredita que o futuro é a não supervisionada e RL" | prediction, not fact |
| L112–114 | Waymo vs. Tesla labelling anecdote | not actionable |
| L128 | "A maior parte das pessoas despreza o primeiro passo" | opinion → only the labelled heuristic is kept |
| L134 | "O mais importante a seguir ao domínio é a qualidade dos dados" | opinion ranking |
| L180 | "pair plot = super power" | opinion; the plot itself is kept in §7 |
| L182 | "com threasholds podemos criar algoritmos…" | garbled; reconstruction would be invention |
| L193 | "NOTA: dispercei no slide antes de Multiple linear regression" | illegible |
| L223 | "o algoritmo é pesado" | vague; no actionable content |
| L236 | "exemplo da poda da cebe" | illegible |
| L246 | "série, paralelo, ou em?" | incomplete; the three standard modes are stated instead |
| L270 | "dá para fazer muita coisa com arquiteturas, é um mundo" | opinion |
| — | "(Calculo da variância ver slide.)" and 5 × "(ver slide)" | placeholders; definitions supplied from standard reference |

## Standard-reference additions `[ref]`

These are not in `notas.md`; they are standard definitions added for correctness.

| Topic | Addition |
|---|---|
| §1 | ordinal rank-distance loss is a choice, not a universal |
| §3 | normal reference values |
| §4 | correlation normalisation and bounds |
| §7 | standardisation default, pair plots |
| §8 | multinomial/one-vs-rest, L1/L2 behaviour |
| §9 | `max_depth`, validation-based stopping, `ccp_alpha`, importance caveat |
| §10 | random-forest feature subsampling, XGBoost second-order/regularisation, `n_estimators` |
| §11 | vanilla network definition, output activations |

## Resolved TODOs

| Original | Resolution |
|---|---|
| L259 "tpc: perceber o que é n_estimators" | answered in §10; removed from open questions |
| L197 "TODO: rever com o chat" | answered in §8; removed |
| L182 thresholds fragment | dropped as unreconstructable; not kept as an open question |
