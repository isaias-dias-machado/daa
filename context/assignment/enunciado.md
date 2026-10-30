<!--
Verbatim body of the assignment PDF (3 pages), with the repeated page numbers and
3 URLs removed. The 5 email addresses are retained as plain text because peer
assessment requires them.
Source PDF: /home/isaias/mestrado/daa/context.bak-2026-10-30/enunciado.md
SHA-256 of source: b576b9e102fb78ce51a93acff43e3e14f19d7f2516840084e87284ae1ec03f77
Transcribed: 2026-10-30.
Do not edit. If the professor issues a new version, re-transcribe and update the hash.
-->

# Group Practical Work

Universidade do Minho

Departamento de Informática

Dados e Aprendizagem Automática

1º Ano, 1º Semestre

Ano letivo 2026/2027

October 2026

## Theme

Design and optimization of Machine Learning models

## Learning Objectives

The aim of this assignment is to encourage and motivate students to design and develop a Machine Learning project using, among others, the learning models covered throughout the semester.

## Work Group

Work groups may be formed by students from different lab classes but must consist of between 2 and 4 members. Students can form groups across any course enrolled in this curricular unit. Please use this link to register your group: [link removed]

## Introduction

The provided dataset originates from a nationwide survey conducted in Portugal during the 2019 calendar year. This study aimed to collect comprehensive information about the resident population, covering dimensions such as self-perceived health status, healthcare utilization, chronic conditions, autonomy, determinants and lifestyles (physical exercise, alcohol and tobacco consumption, diet), as well as general socioeconomic characteristics.

The primary objective of this project is to develop and optimize Machine Learning models capable of predicting an individual's age group based on their biological, behavioral and socioeconomic profile. Furthermore, the model's predictions will be used to detect profile inconsistencies and epidemiological anomalies — identifying cases where an individual's declared age diverges significantly from their biological and lifestyle profile (e.g., a 60-year-old biological profile in an 18-year-old individual, or vice versa) — and to perform a detailed characterization of each age group at both health and socioeconomic levels.

## Assignment

The assignment includes 2 TASKS.

### ANALYSIS AND VALIDATION TASK:

- Explore, analyze and prepare the dataset provided, seeking to extract relevant knowledge in the context of the problem in question;
- Obtain and critically analyze the modelling results;
- Perform a thorough feature engineering process and analyze the factors that contribute the most to characterize biological age and profile anomalies. This can be achieved using model interpretability techniques.

### COMPETITION TASK:

- The groups will have to work on the dataset available at [link removed]
  - The link above redirects to the Kaggle platform where a competition has been created. The dataset to be used, as well as its specifications and operations, are available at that link;
  - The first step is to access the Kaggle platform using the following link to register for the competition: [link removed]
    Students must then form teams with the other members of their working group. The team’s name must follow the format GRUPO_<CURSO(s)>_<X>, where <CURSO(s)> corresponds to the master's program(s) (MEI, MECD, MES, MIA, MMC, MCS, MCA or MRC) and <X> to the group number. Submissions cannot be made on the Kaggle platform if the group is incomplete;
- Design and optimization of Machine Learning models for the competition dataset:
  - Students must submit the results obtained on the Kaggle platform to obtain the model's Macro F1-Score.
  - There is a daily limit of 3 valid submissions, so students should start submitting as soon as possible. The competition closes at the end of January 1, 2027;
- Obtain and critically analyze the results;
- Interpret the results acquired and define their usefulness in the context of the problem underlying the dataset worked on. Determine and explain the most relevant results.

## Delivery and Assessment

The results obtained should be the subject of a report, limited to 20 pages, which presents, among other things:

- The areas to be tackled, the objectives and how it is proposed to achieve them;
- The methodology used and how it was applied;
- A detailed description and exploration of the dataset, as well as any processing carried out;
- A description of the models developed, including their characteristics, how and on which hyperparameters the models were tuned, training characteristics, among other details that should be provided;
- A summary of the results obtained and their critical analysis;
- The identification and characterization of the main profile anomalies detected (cases with large discrepancies between real age and predicted age);
- The presentation of suggestions and recommendations after analyzing the results obtained and the models developed.

The whole process must be accompanied by examples and indications that make it possible to reproduce all the steps taken and the results obtained.

During the class period on November 30 and December 3, 2026, a checkpoint will be made on the work carried out by the working groups, with each group using the means it considers most appropriate to demonstrate the results obtained.

On January 11, 12 and 13, 2027, there will be presentation sessions for the work carried out in both TASKS. The working groups will have to choose the slot they want for their presentation, which will be made available in the coming weeks. Each group will have 10 minutes to make their presentation, using whatever means they deem most appropriate.

The report, as well as all other elements produced, must be compressed into a single zip file which must be submitted by a member of the group by January 2, 2027 on the University of Minho's e-learning platform (under “Conteúdo/Instrumentos de Avaliação em Grupo/Submissão TPG”).

## Peer Assessment

Each group must carry out a collective analysis of the contribution and effort made by each member toward the progress of the project. Based on this analysis, the group should be able to identify which members worked above, at or below the average. This assessment component accounts for 1 value (point) of each student’s grade, reflecting their individual contribution.

To submit this assessment, one member of the group must send an email, with all the other members of the group in CC, to valves@di.uminho.pt, filipa.ferraz@di.uminho.pt, dad@di.uminho.pt, fm@di.uminho.pt and gjfbarbosa@di.uminho.pt. The subject must be "AP DAA - GRUPO_<CURSO(s)>_<X> - Avaliação Por Pares".

In the body of the email, the group must indicate each member’s delta (the value to be added or subtracted from their mark for this component). Please remember that deltas can be negative, zero or positive. Within each group, the sum of the deltas must equal 0.00 and an individual delta can never exceed +1.0 or be less than -1.0.

Example 1 (corresponds to an equal effort by all members):

PG1234 João DELTA = 0
PG5678 António DELTA = 0
PG9123 Maria DELTA = 0
PG4567 Rita DELTA = 0

Example 2 (António receives 1 additional value, Rita keeps her baseline grade, while João and Maria are deducted 0.5 values each):

PG1234 João DELTA = -0.5
PG5678 António DELTA = 1
PG9123 Maria DELTA = -0.5
PG4567 Rita DELTA = 0

## Code of Conduct

The participants in this academic work declare that they have acted with integrity and confirm that they have not resorted to the practice of plagiarism or any form of misuse or falsification of information and results during any of the stages of this project. They also declare that they are fully aware of and have respected the University of Minho's Code of Ethical Conduct.
