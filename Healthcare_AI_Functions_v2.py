# Databricks notebook source
# MAGIC %md
# MAGIC <img src="./databricks-academy.png" alt="Databricks Academy" height="55"/>
# MAGIC
# MAGIC # Hands-On: AI Functions em Healthcare
# MAGIC
# MAGIC ### IA Generativa
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## O que você vai aprender
# MAGIC
# MAGIC Neste notebook, você vai usar **AI Functions nativas do Databricks** — funções SQL que invocam LLMs
# MAGIC diretamente de dentro de uma query — para transformar dados hospitalares brutos em
# MAGIC **inteligência acionável**:
# MAGIC
# MAGIC - Classificar a **prioridade clínica** de cada consulta
# MAGIC - Detectar **sentimento** em avaliações de pacientes
# MAGIC - **Extrair entidades** (sintomas, medicamentos) de texto livre
# MAGIC - Gerar **resumos executivos** automáticos para a diretoria
# MAGIC - Criar **recomendações operacionais** baseadas em padrões
# MAGIC
# MAGIC > **Pré-requisito:** Este notebook assume que as tabelas `healthcare_lakehouse_${sufixo}.bronze.hospitais`,
# MAGIC > `healthcare_lakehouse_${sufixo}.bronze.pacientes` e `healthcare_lakehouse_${sufixo}.bronze.consultas` já existem
# MAGIC > (criadas no Módulo 1 e no Lakeflow Designer).
# MAGIC
# MAGIC ## Roadmap deste notebook
# MAGIC
# MAGIC | Parte | Tema | AI Function |
# MAGIC |-------|------|-------------|
# MAGIC | 1 | Introdução às AI Functions | — |
# MAGIC | 2 | Explorando os dados existentes | — |
# MAGIC | 3 | Primeiro exemplo: classificação de prioridade | `ai_classify` |
# MAGIC | 4 | Análise de sentimento | `ai_analyze_sentiment` |
# MAGIC | 5 | Extração de insights operacionais | `ai_gen` |
# MAGIC | 6 | Classificação inteligente de risco | `ai_classify` + `ai_extract` |
# MAGIC | 7 | Geração de resumos executivos | `ai_summarize` + `ai_gen` |
# MAGIC | 8 | AI/BI + Dashboards | — |
# MAGIC | 9 | Boas práticas, custos e governança | — |
# MAGIC | 10 | Encerramento — o futuro de GenAI em saúde | — |
# MAGIC
# MAGIC > **Compute:** Este notebook roda em **SQL Warehouse Serverless** ou **Serverless Compute** —
# MAGIC > AI Functions são chamadas nativas e não precisam de configuração adicional de endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC # Parte 1 — Introdução às AI Functions
# MAGIC
# MAGIC ## O que são AI Functions?
# MAGIC
# MAGIC AI Functions são **funções SQL nativas do Databricks** que invocam modelos de linguagem (LLMs)
# MAGIC diretamente dentro da sua query. Você usa SQL — não precisa saber Python, não precisa de notebook,
# MAGIC não precisa configurar endpoint.
# MAGIC
# MAGIC ```sql
# MAGIC SELECT ai_analyze_sentiment('O atendimento foi excelente') AS sentimento;
# MAGIC -- → positive
# MAGIC ```
# MAGIC
# MAGIC É isso. Uma linha de SQL → análise de sentimento via LLM → resultado direto na sua tabela.
# MAGIC
# MAGIC ## Como funcionam por dentro
# MAGIC
# MAGIC O fluxo é simples e totalmente gerenciado pelo Databricks:
# MAGIC
# MAGIC 1. **SQL Query** com `ai_classify`, `ai_analyze_sentiment`, `ai_gen` etc.
# MAGIC 2. **Databricks SQL Engine** interpreta a função
# MAGIC 3. **Foundation Model Serving** (Llama, DBRX, GTE) recebe a chamada
# MAGIC 4. **Resultado** retorna como uma coluna SQL comum, pronta para usar em joins, filtros e agregações
# MAGIC
# MAGIC ## IA Tradicional vs. AI Functions (GenAI)
# MAGIC
# MAGIC | Aspecto | IA Tradicional (ML clássico) | AI Functions (GenAI) |
# MAGIC |---------|------------------------------|---------------------|
# MAGIC | **Quem cria** | Cientistas de dados | Analistas, engenheiros, qualquer um com SQL |
# MAGIC | **Treinamento** | Semanas/meses, requer dataset rotulado | **Zero treinamento** — modelo pré-treinado |
# MAGIC | **Deploy** | MLflow, endpoints, monitoramento | **Já está pronto** — chamada SQL |
# MAGIC | **Casos típicos** | Predição numérica, regressão | Texto, classificação, extração, geração |
# MAGIC | **Tempo até valor** | Meses | **Minutos** |
# MAGIC
# MAGIC ## Por que democratizam IA
# MAGIC
# MAGIC Antes das AI Functions, usar IA em saúde exigia:
# MAGIC - Time de ML dedicado
# MAGIC - Pipeline de treinamento e deploy
# MAGIC - Endpoints customizados
# MAGIC - Monitoramento de modelo
# MAGIC
# MAGIC Com AI Functions, **um analista clínico ou de qualidade hospitalar** consegue, em SQL:
# MAGIC - Classificar 1 milhão de consultas em prioridades em uma única query
# MAGIC - Identificar pacientes insatisfeitos automaticamente
# MAGIC - Extrair informações clínicas estruturadas de prontuários em texto livre
# MAGIC
# MAGIC ## Casos de uso em healthcare
# MAGIC
# MAGIC | Área | AI Function | Aplicação |
# MAGIC |------|-------------|-----------|
# MAGIC | **Patient Experience** | `ai_analyze_sentiment` | NPS automático em avaliações |
# MAGIC | **Triagem** | `ai_classify` | Priorização baseada em sintomas |
# MAGIC | **Prontuário** | `ai_extract` | Extrair medicamentos, alergias, diagnósticos |
# MAGIC | **Gestão** | `ai_summarize` | Resumo diário de operação |
# MAGIC | **Compliance** | `ai_mask` | Anonimização de PII/PHI para LGPD/HIPAA |
# MAGIC | **Comunicação** | `ai_translate` | Tradução de prontuários internacionais |
# MAGIC | **Insights** | `ai_gen` | Recomendações operacionais automáticas |

# COMMAND ----------

# MAGIC %md
# MAGIC ## AI Functions disponíveis no Databricks
# MAGIC
# MAGIC | Função | O que faz | Exemplo de retorno |
# MAGIC |--------|-----------|--------------------|
# MAGIC | `ai_analyze_sentiment(text)` | Detecta sentimento | `'positive'`, `'negative'`, `'neutral'`, `'mixed'` |
# MAGIC | `ai_classify(text, ARRAY)` | Classifica em categorias dadas | uma das categorias |
# MAGIC | `ai_extract(text, ARRAY)` | Extrai campos como JSON | `{"sintoma": "dor", "duracao": "3 dias"}` |
# MAGIC | `ai_summarize(text, max_words)` | Resume texto | texto resumido |
# MAGIC | `ai_gen(prompt)` | Geração livre | texto gerado pelo LLM |
# MAGIC | `ai_translate(text, lang)` | Tradução | texto traduzido |
# MAGIC | `ai_fix_grammar(text)` | Corrige gramática | texto corrigido |
# MAGIC | `ai_mask(text, ARRAY)` | Mascara PII/PHI | texto anonimizado |
# MAGIC | `ai_similarity(t1, t2)` | Similaridade semântica | número entre 0 e 1 |
# MAGIC
# MAGIC Todas funcionam direto em SQL, sem instalação, sem configuração.

# COMMAND ----------

# MAGIC %md
# MAGIC # Parte 2 — Explorando os dados existentes
# MAGIC
# MAGIC ### Objetivo
# MAGIC Conectar ao catálogo já criado e explorar as três tabelas que serão a matéria-prima para as AI Functions.
# MAGIC
# MAGIC ### O que vamos olhar
# MAGIC 1. **Schemas** das tabelas (estrutura, tipos)
# MAGIC 2. **Amostras** de dados (sanity check)
# MAGIC 3. **Volumetria** (contagens)
# MAGIC 4. **Relacionamentos** entre as tabelas (visão 360 do paciente)
# MAGIC
# MAGIC ### Conceito: Visão 360 do Paciente
# MAGIC ```
# MAGIC          ┌──────────────┐
# MAGIC          │   PACIENTES  │  ← demografia, plano de saúde, histórico
# MAGIC          └──────┬───────┘
# MAGIC                 │
# MAGIC                 │ id_paciente
# MAGIC                 ▼
# MAGIC          ┌──────────────┐
# MAGIC          │   CONSULTAS  │  ← fato transacional (atendimentos)
# MAGIC          └──────┬───────┘
# MAGIC                 │
# MAGIC                 │ id_hospital
# MAGIC                 ▼
# MAGIC          ┌──────────────┐
# MAGIC          │   HOSPITAIS  │  ← dimensão geográfica/operacional
# MAGIC          └──────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parametrização — Sufixo do Catálogo
# MAGIC
# MAGIC Este notebook lê dados de um catálogo nomeado `healthcare_lakehouse_<sufixo>`.
# MAGIC O widget abaixo pergunta qual sufixo usar — deve ser **o mesmo** que você usou no `Healthcare_Lakehouse_Demo_v2`.

# COMMAND ----------

# DBTITLE 1,Criar widget de sufixo
# MAGIC %sql
# MAGIC CREATE WIDGET TEXT sufixo DEFAULT "leandro";

# COMMAND ----------

# DBTITLE 1,Montar nome do catálogo
# MAGIC %sql
# MAGIC DECLARE OR REPLACE VARIABLE catalog_name STRING;
# MAGIC SET VAR catalog_name = CONCAT('healthcare_lakehouse_', :sufixo);
# MAGIC SELECT catalog_name AS catalogo_em_uso;

# COMMAND ----------

# DBTITLE 1,Selecionar o catálogo
# MAGIC %sql
# MAGIC USE CATALOG IDENTIFIER(catalog_name);
# MAGIC USE SCHEMA bronze;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Garantindo que `consultas` existe
# MAGIC
# MAGIC Se você ainda **não rodou** o pipeline do Lakeflow Designer que materializa `consultas` em Bronze,
# MAGIC a célula abaixo cria uma **tabela temporária `consultas_demo`** com dados sintéticos coerentes
# MAGIC (cruzando `hospitais` e `pacientes` reais) só para este workshop rodar ponta a ponta.
# MAGIC
# MAGIC > **Em produção:** ignore esta célula — use a tabela `consultas` real que vem do Lakeflow.
# MAGIC > Para fins deste hands-on, criamos uma **view** `consultas` que aponta para `consultas_demo`
# MAGIC > caso a tabela real não exista ainda.

# COMMAND ----------

# DBTITLE 1,Criar consultas_demo se não houver consultas
# MAGIC %sql
# MAGIC -- Dados sintéticos coerentes para demo das AI Functions.
# MAGIC -- Usa pmod() para escolha determinística → mesmo resultado a cada execução.
# MAGIC CREATE OR REPLACE TABLE healthcare_lakehouse_${sufixo}.bronze.consultas_demo AS
# MAGIC WITH pacientes_amostra AS (
# MAGIC     SELECT id_paciente, row_number() OVER (ORDER BY id_paciente) AS rn
# MAGIC     FROM healthcare_lakehouse_${sufixo}.bronze.pacientes
# MAGIC     WHERE id_paciente IS NOT NULL
# MAGIC     LIMIT 200
# MAGIC ),
# MAGIC hospitais_idx AS (
# MAGIC     SELECT id_hospital, row_number() OVER (ORDER BY id_hospital) - 1 AS idx,
# MAGIC            COUNT(*) OVER () AS total
# MAGIC     FROM healthcare_lakehouse_${sufixo}.bronze.hospitais
# MAGIC )
# MAGIC SELECT
# MAGIC     p.rn AS id_consulta,
# MAGIC     p.id_paciente,
# MAGIC     h.id_hospital,
# MAGIC     CASE pmod(p.id_paciente, 8)
# MAGIC         WHEN 0 THEN 'Cardiologia'
# MAGIC         WHEN 1 THEN 'Ortopedia'
# MAGIC         WHEN 2 THEN 'Pediatria'
# MAGIC         WHEN 3 THEN 'Clínica Geral'
# MAGIC         WHEN 4 THEN 'Ginecologia'
# MAGIC         WHEN 5 THEN 'Dermatologia'
# MAGIC         WHEN 6 THEN 'Neurologia'
# MAGIC         ELSE 'Oftalmologia'
# MAGIC     END AS especialidade,
# MAGIC     CAST(15 + pmod(p.id_paciente * 11, 80) AS INT) AS tempo_espera_minutos,
# MAGIC     CASE WHEN pmod(p.id_paciente, 10) = 0 THEN 'cancelada' ELSE 'finalizada' END AS status,
# MAGIC     CASE pmod(p.id_paciente, 12)
# MAGIC         WHEN 0 THEN 'Paciente com dor torácica intensa há 30 minutos, sudorese e falta de ar'
# MAGIC         WHEN 1 THEN 'Dor lombar há 2 semanas, piora ao se movimentar'
# MAGIC         WHEN 2 THEN 'Febre alta de 39°C há 3 dias com tosse seca'
# MAGIC         WHEN 3 THEN 'Manchas vermelhas pelo corpo, coceira intensa há 1 semana'
# MAGIC         WHEN 4 THEN 'Cefaleia recorrente há 1 mês, tonturas ocasionais'
# MAGIC         WHEN 5 THEN 'Dor abdominal aguda, náuseas e vômito há 12 horas'
# MAGIC         WHEN 6 THEN 'Visão embaçada há 2 dias, sem outros sintomas'
# MAGIC         WHEN 7 THEN 'Acompanhamento de rotina, sem queixas específicas'
# MAGIC         WHEN 8 THEN 'Hipertensão arterial descontrolada, pressão 18x12 medida em casa'
# MAGIC         WHEN 9 THEN 'Falta de ar progressiva há 1 semana, edema em membros inferiores'
# MAGIC         WHEN 10 THEN 'Dor de garganta há 2 dias, ardor ao engolir'
# MAGIC         ELSE 'Check-up anual preventivo'
# MAGIC     END AS sintomas_relatados
# MAGIC FROM pacientes_amostra p
# MAGIC JOIN hospitais_idx h ON h.idx = pmod(p.id_paciente * 7, h.total);

# COMMAND ----------

# DBTITLE 1,View consultas (usa real se existir, senão demo)
# MAGIC %sql
# MAGIC -- Cria view "consultas" apontando para a tabela demo.
# MAGIC -- Se você já tem bronze.consultas real materializada pelo Lakeflow,
# MAGIC -- substitua esta view por: CREATE OR REPLACE VIEW consultas AS SELECT * FROM consultas_real
# MAGIC CREATE OR REPLACE VIEW healthcare_lakehouse_${sufixo}.bronze.consultas AS
# MAGIC SELECT * FROM healthcare_lakehouse_${sufixo}.bronze.consultas_demo;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tabela 1 — Hospitais
# MAGIC Dimensão geográfica e operacional. Cada consulta acontece em um hospital.

# COMMAND ----------

# DBTITLE 1,Schema e amostra — hospitais
# MAGIC %sql
# MAGIC DESCRIBE TABLE hospitais;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM hospitais LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tabela 2 — Pacientes
# MAGIC Cadastro de pacientes com demografia, contato e plano de saúde. **Contém PII** — em produção deve ter
# MAGIC políticas de mascaramento aplicadas via Unity Catalog.

# COMMAND ----------

# DBTITLE 1,Schema e amostra — pacientes
# MAGIC %sql
# MAGIC DESCRIBE TABLE pacientes;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM pacientes LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tabela 3 — Consultas
# MAGIC Fato transacional. Cada linha é um atendimento realizado, com sintomas relatados, diagnóstico,
# MAGIC tempo de espera e modalidade.

# COMMAND ----------

# DBTITLE 1,Schema e amostra — consultas
# MAGIC %sql
# MAGIC DESCRIBE TABLE consultas;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM consultas LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Volumetria do ambiente

# COMMAND ----------

# DBTITLE 1,Contagem de registros
# MAGIC %sql
# MAGIC SELECT 'hospitais' AS tabela, COUNT(*) AS total FROM hospitais
# MAGIC UNION ALL
# MAGIC SELECT 'pacientes', COUNT(*) FROM pacientes
# MAGIC UNION ALL
# MAGIC SELECT 'consultas', COUNT(*) FROM consultas;

# COMMAND ----------

# MAGIC %md
# MAGIC # Parte 3 — Primeiro exemplo: classificação de prioridade
# MAGIC
# MAGIC ### Objetivo de negócio
# MAGIC Toda manhã o hospital recebe centenas de consultas marcadas. **Qual delas é urgente? Qual é eletiva?**
# MAGIC Hoje, essa triagem é manual e demora. Vamos automatizar com IA.
# MAGIC
# MAGIC ### Conceito: `ai_classify`
# MAGIC `ai_classify(texto, ARRAY('categoria1', 'categoria2', ...))` força o LLM a escolher **uma** das
# MAGIC categorias que você definiu. É **classificação supervisionada zero-shot** — você não treinou nada,
# MAGIC apenas descreveu as opções.
# MAGIC
# MAGIC ### Prompt engineering implícito
# MAGIC O Databricks monta o prompt por baixo dos panos, algo como:
# MAGIC > "Classifique o texto a seguir em UMA das categorias: [urgente, alta, média, eletiva]. Texto: {sintomas}"
# MAGIC
# MAGIC ### Resultado esperado
# MAGIC Uma nova coluna `prioridade_ia` em cada consulta, com a classificação feita pelo LLM.

# COMMAND ----------

# DBTITLE 1,Exemplo simples — uma única consulta
# MAGIC %sql
# MAGIC SELECT
# MAGIC     'Paciente com dor torácica intensa há 30 minutos, sudorese e falta de ar' AS sintomas,
# MAGIC     ai_classify(
# MAGIC         'Paciente com dor torácica intensa há 30 minutos, sudorese e falta de ar',
# MAGIC         ARRAY('urgente', 'alta', 'media', 'eletiva')
# MAGIC     ) AS prioridade_ia;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agora em escala — classificando todas as consultas
# MAGIC
# MAGIC O mesmo `ai_classify` aplicado a **toda a tabela**. O LLM é chamado linha a linha,
# MAGIC mas o Databricks paraleliza automaticamente.

# COMMAND ----------

# DBTITLE 1,Classificação de prioridade em escala
# MAGIC %sql
# MAGIC SELECT
# MAGIC     id_consulta,
# MAGIC     especialidade,
# MAGIC     sintomas_relatados,
# MAGIC     ai_classify(
# MAGIC         sintomas_relatados,
# MAGIC         ARRAY('urgente', 'alta', 'media', 'eletiva')
# MAGIC     ) AS prioridade_ia
# MAGIC FROM consultas
# MAGIC WHERE sintomas_relatados IS NOT NULL
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribuição da prioridade no hospital
# MAGIC
# MAGIC Materializando o resultado e contando — agora temos uma **visão operacional automática** da fila.

# COMMAND ----------

# DBTITLE 1,Distribuição de prioridades
# MAGIC %sql
# MAGIC WITH consultas_classificadas AS (
# MAGIC     SELECT
# MAGIC         id_consulta,
# MAGIC         especialidade,
# MAGIC         ai_classify(
# MAGIC             sintomas_relatados,
# MAGIC             ARRAY('urgente', 'alta', 'media', 'eletiva')
# MAGIC         ) AS prioridade_ia
# MAGIC     FROM consultas
# MAGIC     WHERE sintomas_relatados IS NOT NULL
# MAGIC     LIMIT 100  -- limite para a demo, em produção remova
# MAGIC )
# MAGIC SELECT
# MAGIC     prioridade_ia,
# MAGIC     COUNT(*) AS total_consultas,
# MAGIC     ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentual
# MAGIC FROM consultas_classificadas
# MAGIC GROUP BY prioridade_ia
# MAGIC ORDER BY total_consultas DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Valor de negócio
# MAGIC - **Antes:** triagem manual, fila por ordem de chegada, risco de atender quadro grave depois de eletivo.
# MAGIC - **Depois:** priorização automática em segundos, equipe focada nos casos urgentes primeiro.
# MAGIC - **Impacto:** redução de tempo até atendimento crítico, melhoria de outcomes clínicos, otimização de recursos.

# COMMAND ----------

# MAGIC %md
# MAGIC # Parte 4 — Análise de sentimento em avaliações
# MAGIC
# MAGIC ### Contexto: Patient Experience
# MAGIC Hospitais coletam **milhares de avaliações** por mês — formulários, NPS, comentários no app.
# MAGIC Ler tudo manualmente é impossível. Vamos usar `ai_analyze_sentiment` para classificar todas
# MAGIC em segundos.
# MAGIC
# MAGIC ### Conceito: `ai_analyze_sentiment`
# MAGIC Retorna uma das categorias: `'positive'`, `'negative'`, `'neutral'`, `'mixed'`.
# MAGIC
# MAGIC > **Nota:** Este notebook simula avaliações fictícias inline (sem depender de tabela externa)
# MAGIC > para que você possa rodar mesmo se ainda não criou a tabela de avaliações no Lakeflow Designer.

# COMMAND ----------

# DBTITLE 1,Sentimento em comentários simulados de pacientes
# MAGIC %sql
# MAGIC WITH avaliacoes_simuladas AS (
# MAGIC     SELECT * FROM (VALUES
# MAGIC         (1, 'Atendimento excelente e rápido, equipe muito atenciosa'),
# MAGIC         (2, 'Demorei mais de 2 horas para ser atendido, péssimo'),
# MAGIC         (3, 'Médico extremamente atencioso, explicou tudo com calma'),
# MAGIC         (4, 'Recepção desorganizada, ninguém sabia me informar'),
# MAGIC         (5, 'Hospital limpo, mas demora demais na espera'),
# MAGIC         (6, 'Voltei a me sentir bem após a consulta, recomendo'),
# MAGIC         (7, 'Estacionamento caro, mas o atendimento valeu a pena'),
# MAGIC         (8, 'Não fui bem tratado pela enfermeira, falta de empatia'),
# MAGIC         (9, 'Consulta dentro do horário, médico competente'),
# MAGIC         (10, 'Sistema de marcação online não funciona direito')
# MAGIC     ) AS t(id_avaliacao, comentario)
# MAGIC )
# MAGIC SELECT
# MAGIC     id_avaliacao,
# MAGIC     comentario,
# MAGIC     ai_analyze_sentiment(comentario) AS sentimento
# MAGIC FROM avaliacoes_simuladas
# MAGIC ORDER BY id_avaliacao;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Combinando sentimento com classificação de tema
# MAGIC
# MAGIC `ai_classify` complementa o sentimento — sabemos **o que** o paciente comentou,
# MAGIC além de **como** ele se sentiu.

# COMMAND ----------

# DBTITLE 1,Sentimento + tema da avaliação
# MAGIC %sql
# MAGIC WITH avaliacoes_simuladas AS (
# MAGIC     SELECT * FROM (VALUES
# MAGIC         (1, 'Atendimento excelente e rápido, equipe muito atenciosa'),
# MAGIC         (2, 'Demorei mais de 2 horas para ser atendido, péssimo'),
# MAGIC         (3, 'Médico extremamente atencioso, explicou tudo com calma'),
# MAGIC         (4, 'Recepção desorganizada, ninguém sabia me informar'),
# MAGIC         (5, 'Hospital limpo, mas demora demais na espera'),
# MAGIC         (6, 'Voltei a me sentir bem após a consulta, recomendo'),
# MAGIC         (7, 'Estacionamento caro, mas o atendimento valeu a pena'),
# MAGIC         (8, 'Não fui bem tratado pela enfermeira, falta de empatia'),
# MAGIC         (9, 'Consulta dentro do horário, médico competente'),
# MAGIC         (10, 'Sistema de marcação online não funciona direito')
# MAGIC     ) AS t(id_avaliacao, comentario)
# MAGIC )
# MAGIC SELECT
# MAGIC     id_avaliacao,
# MAGIC     comentario,
# MAGIC     ai_analyze_sentiment(comentario) AS sentimento,
# MAGIC     ai_classify(
# MAGIC         comentario,
# MAGIC         ARRAY('tempo_de_espera', 'qualidade_medica', 'atendimento_recepcao', 'infraestrutura', 'sistema_digital', 'outros')
# MAGIC     ) AS tema
# MAGIC FROM avaliacoes_simuladas;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visão executiva — NPS automático

# COMMAND ----------

# DBTITLE 1,NPS calculado automaticamente
# MAGIC %sql
# MAGIC WITH avaliacoes_simuladas AS (
# MAGIC     SELECT * FROM (VALUES
# MAGIC         (1, 'Atendimento excelente e rápido, equipe muito atenciosa'),
# MAGIC         (2, 'Demorei mais de 2 horas para ser atendido, péssimo'),
# MAGIC         (3, 'Médico extremamente atencioso, explicou tudo com calma'),
# MAGIC         (4, 'Recepção desorganizada, ninguém sabia me informar'),
# MAGIC         (5, 'Hospital limpo, mas demora demais na espera'),
# MAGIC         (6, 'Voltei a me sentir bem após a consulta, recomendo'),
# MAGIC         (7, 'Estacionamento caro, mas o atendimento valeu a pena'),
# MAGIC         (8, 'Não fui bem tratado pela enfermeira, falta de empatia'),
# MAGIC         (9, 'Consulta dentro do horário, médico competente'),
# MAGIC         (10, 'Sistema de marcação online não funciona direito')
# MAGIC     ) AS t(id_avaliacao, comentario)
# MAGIC ),
# MAGIC com_sentimento AS (
# MAGIC     SELECT
# MAGIC         id_avaliacao,
# MAGIC         ai_analyze_sentiment(comentario) AS sentimento
# MAGIC     FROM avaliacoes_simuladas
# MAGIC )
# MAGIC SELECT
# MAGIC     sentimento,
# MAGIC     COUNT(*) AS total,
# MAGIC     ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentual
# MAGIC FROM com_sentimento
# MAGIC GROUP BY sentimento
# MAGIC ORDER BY total DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Valor de negócio
# MAGIC - **Visão em tempo real** da satisfação do paciente
# MAGIC - **Identificação proativa** de áreas com problemas recorrentes (ex: recepção, tempo de espera)
# MAGIC - **Ação direcionada** — gestor sabe exatamente onde investir

# COMMAND ----------

# MAGIC %md
# MAGIC # Parte 5 — Extração de insights operacionais
# MAGIC
# MAGIC ### Objetivo
# MAGIC Até agora, classificamos e analisamos dados estruturados. Agora vamos pedir ao LLM para
# MAGIC **gerar insights em linguagem natural** sobre métricas operacionais.
# MAGIC
# MAGIC ### Conceito: `ai_gen`
# MAGIC `ai_gen(prompt)` é geração livre — você passa um prompt e o LLM responde em texto.
# MAGIC Útil para resumos, explicações, recomendações.
# MAGIC
# MAGIC ### Estratégia
# MAGIC 1. Calcular métricas com SQL tradicional
# MAGIC 2. Concatenar as métricas num prompt
# MAGIC 3. Pedir ao LLM para gerar uma análise executiva

# COMMAND ----------

# DBTITLE 1,Métricas operacionais por especialidade
# MAGIC %sql
# MAGIC SELECT
# MAGIC     especialidade,
# MAGIC     COUNT(*) AS total_consultas,
# MAGIC     ROUND(AVG(tempo_espera_minutos), 1) AS tempo_espera_medio,
# MAGIC     ROUND(AVG(CASE WHEN status = 'finalizada' THEN 1.0 ELSE 0.0 END) * 100, 1) AS taxa_conclusao_pct
# MAGIC FROM consultas
# MAGIC GROUP BY especialidade
# MAGIC ORDER BY tempo_espera_medio DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pedindo ao LLM para gerar análise executiva
# MAGIC
# MAGIC Note como concatenamos as métricas num único texto e pedimos uma interpretação.

# COMMAND ----------

# DBTITLE 1,Insight automático gerado por IA
# MAGIC %sql
# MAGIC WITH metricas AS (
# MAGIC     SELECT
# MAGIC         especialidade,
# MAGIC         COUNT(*) AS total_consultas,
# MAGIC         ROUND(AVG(tempo_espera_minutos), 1) AS tempo_espera_medio
# MAGIC     FROM consultas
# MAGIC     GROUP BY especialidade
# MAGIC ),
# MAGIC resumo_texto AS (
# MAGIC     SELECT CONCAT_WS(
# MAGIC         '. ',
# MAGIC         COLLECT_LIST(
# MAGIC             CONCAT(especialidade, ': ', total_consultas, ' consultas, espera média ', tempo_espera_medio, ' min')
# MAGIC         )
# MAGIC     ) AS texto_metricas
# MAGIC     FROM metricas
# MAGIC )
# MAGIC SELECT
# MAGIC     ai_gen(
# MAGIC         CONCAT(
# MAGIC             'Você é um diretor de operações hospitalares. Analise estas métricas e gere um parágrafo executivo ',
# MAGIC             'em português brasileiro identificando o principal gargalo e sugerindo uma ação concreta. Métricas: ',
# MAGIC             texto_metricas
# MAGIC         )
# MAGIC     ) AS analise_executiva
# MAGIC FROM resumo_texto;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recomendações por especialidade
# MAGIC
# MAGIC Cada especialidade gera sua própria recomendação personalizada.

# COMMAND ----------

# DBTITLE 1,Recomendações automáticas por especialidade
# MAGIC %sql
# MAGIC WITH metricas AS (
# MAGIC     SELECT
# MAGIC         especialidade,
# MAGIC         COUNT(*) AS total_consultas,
# MAGIC         ROUND(AVG(tempo_espera_minutos), 1) AS tempo_espera_medio
# MAGIC     FROM consultas
# MAGIC     GROUP BY especialidade
# MAGIC )
# MAGIC SELECT
# MAGIC     especialidade,
# MAGIC     total_consultas,
# MAGIC     tempo_espera_medio,
# MAGIC     ai_gen(
# MAGIC         CONCAT(
# MAGIC             'Como diretor hospitalar, dê UMA recomendação curta (máx 30 palavras) em português ',
# MAGIC             'para a especialidade ', especialidade, ' que teve ', total_consultas,
# MAGIC             ' consultas com espera média de ', tempo_espera_medio, ' minutos.'
# MAGIC         )
# MAGIC     ) AS recomendacao_ia
# MAGIC FROM metricas
# MAGIC ORDER BY tempo_espera_medio DESC
# MAGIC LIMIT 5;

# COMMAND ----------

# MAGIC %md
# MAGIC # Parte 6 — Classificação inteligente de risco
# MAGIC
# MAGIC ### Objetivo
# MAGIC Combinar **classificação** (`ai_classify`) com **extração estruturada** (`ai_extract`) para
# MAGIC criar um score de risco do paciente baseado em sintomas relatados.
# MAGIC
# MAGIC ### Conceito: `ai_extract`
# MAGIC `ai_extract(texto, ARRAY('campo1', 'campo2', ...))` extrai os campos solicitados do texto
# MAGIC e retorna como **struct JSON** — perfeito para alimentar tabelas estruturadas.
# MAGIC
# MAGIC ### Caso de uso clínico
# MAGIC A partir de sintomas em texto livre, vamos extrair:
# MAGIC - **Sintoma principal**
# MAGIC - **Duração**
# MAGIC - **Intensidade**
# MAGIC
# MAGIC E então classificar em níveis de risco.

# COMMAND ----------

# DBTITLE 1,Extração de campos clínicos dos sintomas
# MAGIC %sql
# MAGIC SELECT
# MAGIC     id_consulta,
# MAGIC     especialidade,
# MAGIC     sintomas_relatados,
# MAGIC     ai_extract(
# MAGIC         sintomas_relatados,
# MAGIC         ARRAY('sintoma_principal', 'duracao', 'intensidade')
# MAGIC     ) AS sintomas_estruturados
# MAGIC FROM consultas
# MAGIC WHERE sintomas_relatados IS NOT NULL
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score de risco — combinando extração e classificação

# COMMAND ----------

# DBTITLE 1,Risco clínico classificado
# MAGIC %sql
# MAGIC SELECT
# MAGIC     id_consulta,
# MAGIC     especialidade,
# MAGIC     sintomas_relatados,
# MAGIC     ai_classify(
# MAGIC         sintomas_relatados,
# MAGIC         ARRAY('baixo_risco', 'medio_risco', 'alto_risco')
# MAGIC     ) AS nivel_risco,
# MAGIC     ai_extract(
# MAGIC         sintomas_relatados,
# MAGIC         ARRAY('sintoma_principal', 'duracao')
# MAGIC     ) AS detalhes
# MAGIC FROM consultas
# MAGIC WHERE sintomas_relatados IS NOT NULL
# MAGIC LIMIT 15;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorização de pacientes (cruzando consultas com cadastro)
# MAGIC
# MAGIC Aqui combinamos dados estruturados (idade, plano) com IA para um perfil completo.

# COMMAND ----------

# DBTITLE 1,Perfil de risco com contexto demográfico
# MAGIC %sql
# MAGIC SELECT
# MAGIC     c.id_consulta,
# MAGIC     p.nome,
# MAGIC     YEAR(CURRENT_DATE()) - YEAR(p.data_nascimento) AS idade,
# MAGIC     p.plano_saude,
# MAGIC     c.especialidade,
# MAGIC     ai_classify(
# MAGIC         c.sintomas_relatados,
# MAGIC         ARRAY('baixo_risco', 'medio_risco', 'alto_risco')
# MAGIC     ) AS nivel_risco_ia
# MAGIC FROM consultas c
# MAGIC INNER JOIN pacientes p ON c.id_paciente = p.id_paciente
# MAGIC WHERE c.sintomas_relatados IS NOT NULL
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Valor de negócio
# MAGIC - **Triagem inteligente** baseada em sintomas livres digitados pelo recepcionista
# MAGIC - **Decision support** ao médico — IA sugere nível de risco antes da consulta
# MAGIC - **Alocação de recursos** — pacientes de alto risco vão para médicos mais experientes ou exames imediatos

# COMMAND ----------

# MAGIC %md
# MAGIC # Parte 7 — Geração de resumos executivos
# MAGIC
# MAGIC ### Conceito: `ai_summarize`
# MAGIC `ai_summarize(texto, max_words)` resume um texto longo respeitando o limite de palavras.
# MAGIC
# MAGIC ### Caso de uso
# MAGIC Toda manhã, a diretoria recebe um briefing. Em vez de um analista escrever manualmente,
# MAGIC vamos gerar automaticamente com IA.

# COMMAND ----------

# DBTITLE 1,Resumo diário operacional automático
# MAGIC %sql
# MAGIC WITH metricas_dia AS (
# MAGIC     SELECT
# MAGIC         COUNT(*) AS total_consultas,
# MAGIC         COUNT(DISTINCT id_hospital) AS hospitais_ativos,
# MAGIC         COUNT(DISTINCT especialidade) AS especialidades,
# MAGIC         ROUND(AVG(tempo_espera_minutos), 1) AS tempo_espera_medio,
# MAGIC         ROUND(AVG(CASE WHEN status = 'finalizada' THEN 1.0 ELSE 0.0 END) * 100, 1) AS taxa_conclusao
# MAGIC     FROM consultas
# MAGIC ),
# MAGIC briefing AS (
# MAGIC     SELECT CONCAT(
# MAGIC         'Operação do dia: ', total_consultas, ' consultas realizadas em ', hospitais_ativos,
# MAGIC         ' hospitais ativos cobrindo ', especialidades, ' especialidades. ',
# MAGIC         'Tempo médio de espera: ', tempo_espera_medio, ' minutos. ',
# MAGIC         'Taxa de conclusão: ', taxa_conclusao, '%.'
# MAGIC     ) AS texto_completo
# MAGIC     FROM metricas_dia
# MAGIC )
# MAGIC SELECT
# MAGIC     texto_completo AS briefing_bruto,
# MAGIC     ai_gen(
# MAGIC         CONCAT(
# MAGIC             'Reescreva o seguinte briefing operacional como um parágrafo executivo elegante ',
# MAGIC             'em português brasileiro, linguagem corporativa, máximo 80 palavras, ',
# MAGIC             'destacando 1 ponto positivo e 1 ponto de atenção: ',
# MAGIC             texto_completo
# MAGIC         )
# MAGIC     ) AS briefing_executivo_ia
# MAGIC FROM briefing;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Resumo por hospital — para reunião com cada diretor

# COMMAND ----------

# DBTITLE 1,Resumo executivo por hospital
# MAGIC %sql
# MAGIC WITH metricas_hospital AS (
# MAGIC     SELECT
# MAGIC         h.nome_hospital,
# MAGIC         h.cidade,
# MAGIC         COUNT(*) AS total_consultas,
# MAGIC         ROUND(AVG(c.tempo_espera_minutos), 1) AS tempo_espera_medio,
# MAGIC         COUNT(DISTINCT c.especialidade) AS especialidades_ativas
# MAGIC     FROM consultas c
# MAGIC     INNER JOIN hospitais h ON c.id_hospital = h.id_hospital
# MAGIC     GROUP BY h.nome_hospital, h.cidade
# MAGIC )
# MAGIC SELECT
# MAGIC     nome_hospital,
# MAGIC     cidade,
# MAGIC     total_consultas,
# MAGIC     tempo_espera_medio,
# MAGIC     ai_gen(
# MAGIC         CONCAT(
# MAGIC             'Gere um resumo executivo de 2 frases em português para o diretor do ', nome_hospital,
# MAGIC             ' (', cidade, '). Métricas: ', total_consultas, ' consultas, ',
# MAGIC             tempo_espera_medio, ' min de espera média, ', especialidades_ativas, ' especialidades ativas.'
# MAGIC         )
# MAGIC     ) AS resumo_executivo
# MAGIC FROM metricas_hospital
# MAGIC ORDER BY total_consultas DESC
# MAGIC LIMIT 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Identificação automática de principais problemas

# COMMAND ----------

# DBTITLE 1,Top problemas detectados pela IA
# MAGIC %sql
# MAGIC WITH problemas_simulados AS (
# MAGIC     SELECT * FROM (VALUES
# MAGIC         ('Cardiologia teve aumento de 40% nas consultas urgentes esta semana'),
# MAGIC         ('Tempo de espera em Ortopedia subiu de 25 para 47 minutos'),
# MAGIC         ('Hospital Santa Casa SP reportou 3 reclamações sobre recepção'),
# MAGIC         ('Especialidade Pediatria com 95% de taxa de conclusão — referência'),
# MAGIC         ('Sistema de marcação online instável entre 9h e 11h')
# MAGIC     ) AS t(observacao)
# MAGIC )
# MAGIC SELECT
# MAGIC     observacao,
# MAGIC     ai_classify(observacao, ARRAY('problema_critico', 'problema_atencao', 'destaque_positivo', 'informativo')) AS criticidade,
# MAGIC     ai_gen(CONCAT('Em uma frase curta, sugira uma ação corretiva para: ', observacao)) AS acao_sugerida
# MAGIC FROM problemas_simulados;

# COMMAND ----------

# MAGIC %md
# MAGIC # Parte 8 — AI/BI + Dashboards
# MAGIC
# MAGIC ## Como AI Functions alimentam outras camadas
# MAGIC
# MAGIC As colunas geradas por AI Functions fluem naturalmente para o restante da plataforma:
# MAGIC
# MAGIC | Camada | Origem | Consumido por |
# MAGIC |--------|--------|---------------|
# MAGIC | **Bronze** | `hospitais`, `pacientes`, `consultas` (dados crus) | AI Functions |
# MAGIC | **Silver + Gold** | Colunas enriquecidas (prioridade, sentimento, risco, resumo) | AI/BI, Genie, Apps |
# MAGIC | **AI/BI Dashboards** | KPIs enriquecidos, visualizações executivas | Diretoria |
# MAGIC | **Genie Spaces** | Q&A em linguagem natural sobre os dados enriquecidos | Gestores, médicos |
# MAGIC | **Databricks Apps** | App de patient experience, copilot clínico, agentes | Pacientes e equipe |
# MAGIC
# MAGIC ### Exemplos de KPIs enriquecidos por IA
# MAGIC
# MAGIC | KPI tradicional | KPI enriquecido por IA |
# MAGIC |-----------------|------------------------|
# MAGIC | Total de consultas | Total de consultas **por nível de prioridade** (`ai_classify`) |
# MAGIC | Tempo médio de espera | Tempo médio **com correlação ao sentimento** (`ai_analyze_sentiment`) |
# MAGIC | Volume de avaliações | Volume **por tema detectado** (`ai_classify`) |
# MAGIC | Especialidades atendidas | Especialidades **com risco médio do paciente** (`ai_classify`) |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Criando uma view Silver pronta para Genie

# COMMAND ----------

# DBTITLE 1,View enriquecida para AI/BI e Genie
# MAGIC %sql
# MAGIC CREATE OR REPLACE VIEW healthcare_lakehouse_${sufixo}.silver.v_consultas_enriquecidas_ia
# MAGIC COMMENT 'Consultas com colunas geradas por AI Functions: prioridade, risco e tema'
# MAGIC AS
# MAGIC SELECT
# MAGIC     c.id_consulta,
# MAGIC     c.id_paciente,
# MAGIC     c.id_hospital,
# MAGIC     c.especialidade,
# MAGIC     c.tempo_espera_minutos,
# MAGIC     c.status,
# MAGIC     c.sintomas_relatados,
# MAGIC     ai_classify(c.sintomas_relatados, ARRAY('urgente', 'alta', 'media', 'eletiva')) AS prioridade_ia,
# MAGIC     ai_classify(c.sintomas_relatados, ARRAY('baixo_risco', 'medio_risco', 'alto_risco')) AS risco_ia
# MAGIC FROM healthcare_lakehouse_${sufixo}.bronze.consultas c
# MAGIC WHERE c.sintomas_relatados IS NOT NULL;

# COMMAND ----------

# MAGIC %md
# MAGIC > **Observação importante:** Em produção, **NÃO** chame AI Functions em views consultadas com
# MAGIC > alta frequência — cada execução custa inferência. O padrão correto é:
# MAGIC > 1. Executar as AI Functions em um **pipeline batch** (Lakeflow / Workflows)
# MAGIC > 2. Materializar o resultado numa tabela Delta
# MAGIC > 3. As views/dashboards leem a tabela materializada (rápido e barato)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Perguntas naturais que o Genie consegue responder
# MAGIC
# MAGIC Com a view enriquecida, um diretor consegue perguntar em português:
# MAGIC
# MAGIC - "Quais hospitais têm mais consultas urgentes?"
# MAGIC - "Qual especialidade concentra os pacientes de alto risco?"
# MAGIC - "Mostre o tempo médio de espera para consultas urgentes vs eletivas"
# MAGIC - "Quais os top 3 hospitais com pacientes de maior risco?"
# MAGIC
# MAGIC E o Genie traduz isso em SQL automaticamente.

# COMMAND ----------

# MAGIC %md
# MAGIC # Parte 9 — Boas práticas, custos e governança
# MAGIC
# MAGIC ## Custos de inferência
# MAGIC
# MAGIC | Item | Como funciona | Boa prática |
# MAGIC |------|---------------|-------------|
# MAGIC | **Cobrança** | Por token processado (entrada + saída) | Use prompts curtos e diretos |
# MAGIC | **Cache** | Resultados não são cacheados automaticamente | Materialize em tabelas Delta após gerar |
# MAGIC | **Volume** | Cada linha = 1 chamada ao LLM | Para alto volume, considere Batch Inference |
# MAGIC | **Modelo** | Default é o pay-per-token serving | Para SLA crítico, configure provisioned throughput |
# MAGIC
# MAGIC ## Qualidade de prompts
# MAGIC
# MAGIC | Princípio | Exemplo ruim | Exemplo bom |
# MAGIC |-----------|--------------|-------------|
# MAGIC | **Seja específico** | "Analise isso" | "Em português, classifique em [A, B, C]" |
# MAGIC | **Defina o formato** | "Resuma" | "Resuma em 1 parágrafo, máximo 50 palavras" |
# MAGIC | **Dê contexto** | "É urgente?" | "Como triagem hospitalar, é caso de emergência?" |
# MAGIC | **Use categorias fechadas** | `ai_gen` para classificar | `ai_classify` com array explícito |
# MAGIC
# MAGIC ## Governança & segurança
# MAGIC
# MAGIC | Tópico | O que o Databricks oferece |
# MAGIC |--------|---------------------------|
# MAGIC | **PII / PHI** | `ai_mask()` para anonimização, Column Masking via UC |
# MAGIC | **LGPD / HIPAA** | Dados não saem do workspace, inferência dentro do perímetro |
# MAGIC | **Auditoria** | Toda chamada registrada no system tables (`system.access.audit`) |
# MAGIC | **Lineage** | Unity Catalog rastreia origem dos dados das colunas IA |
# MAGIC | **Acesso** | Grants em SQL controlam quem pode usar AI Functions |
# MAGIC
# MAGIC ## Monitoramento e observabilidade
# MAGIC
# MAGIC - **Lakehouse Monitoring** detecta drift em colunas geradas por IA
# MAGIC - **Inference Tables** logam todas as inferências do Foundation Model Serving
# MAGIC - Métricas de latência e qualidade são auditáveis

# COMMAND ----------

# DBTITLE 1,Exemplo de mascaramento PII com ai_mask
# MAGIC %sql
# MAGIC SELECT
# MAGIC     'Paciente João Silva, CPF 123.456.789-00, telefone (11) 98765-4321, com dor abdominal' AS texto_original,
# MAGIC     ai_mask(
# MAGIC         'Paciente João Silva, CPF 123.456.789-00, telefone (11) 98765-4321, com dor abdominal',
# MAGIC         ARRAY('person', 'phone', 'national_id')
# MAGIC     ) AS texto_anonimizado;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Limites da IA generativa
# MAGIC
# MAGIC É essencial conhecer os **limites** da tecnologia, especialmente em saúde:
# MAGIC
# MAGIC | Limitação | Como lidar |
# MAGIC |-----------|------------|
# MAGIC | **Alucinação** | LLM pode inventar — sempre tenha humano no loop para decisões clínicas |
# MAGIC | **Viés** | Modelos podem refletir vieses dos dados de treino |
# MAGIC | **Não-determinismo** | Mesmo input pode gerar saídas levemente diferentes |
# MAGIC | **Não substitui médico** | IA é decision support, **não** decisão autônoma em healthcare |
# MAGIC | **Idioma** | Use prompts em português para resultados em português |
# MAGIC
# MAGIC > **Regra de ouro em saúde:** IA Generativa **augmenta** o profissional, nunca o substitui.
# MAGIC > Toda decisão clínica final deve ter validação humana.

# COMMAND ----------

# MAGIC %md
# MAGIC # Parte 10 — Encerramento
# MAGIC
# MAGIC ## O futuro de GenAI em healthcare
# MAGIC
# MAGIC Você acabou de ver como **uma linha de SQL** pode entregar capacidades que, há 2 anos,
# MAGIC exigiam um time de Machine Learning dedicado, semanas de treinamento e infraestrutura complexa.
# MAGIC
# MAGIC Esse é o **novo patamar** da IA em saúde — a **Modern Data Intelligence Platform**
# MAGIC do Databricks combina **Lakehouse + Unity Catalog**, **AI/BI Dashboards**, **Genie Spaces**
# MAGIC e **Intelligent Apps**, todos sustentados pelas **AI Functions** rodando em SQL nativo.
# MAGIC
# MAGIC ## Decision Intelligence em healthcare
# MAGIC
# MAGIC | Camada | Tecnologia | Exemplo hospitalar |
# MAGIC |--------|-----------|---------------------|
# MAGIC | **Dados** | Lakehouse + Unity Catalog | Visão 360 do paciente governada |
# MAGIC | **Inteligência** | AI Functions + Foundation Models | Classificação, sentimento, extração |
# MAGIC | **Interface** | AI/BI Dashboards + Genie | Diretoria consulta em português |
# MAGIC | **Aplicação** | Databricks Apps + Agentes | Copilot clínico, app de paciente |
# MAGIC
# MAGIC ## O que vem pela frente
# MAGIC
# MAGIC - **Agentes autônomos** que monitoram operação 24/7 e disparam alertas
# MAGIC - **Copilots clínicos** integrados ao prontuário, sugerindo diagnósticos
# MAGIC - **Patient Apps** com IA conversacional para acompanhamento pós-consulta
# MAGIC - **Predictive Operations** combinando IA generativa + ML clássico para forecast
# MAGIC - **Multi-modal**: texto + imagem (laudos radiológicos) + áudio (telemedicina)
# MAGIC
# MAGIC ## Próximos passos sugeridos
# MAGIC
# MAGIC | Próximo módulo | O que você fará |
# MAGIC |----------------|------------------|
# MAGIC | **AI/BI Dashboard** | Construir um dashboard usando a view enriquecida criada na Parte 8 |
# MAGIC | **Genie Space** | Configurar Q&A em linguagem natural sobre os dados |
# MAGIC | **Lakeflow Designer** | Materializar as colunas IA em pipeline batch |
# MAGIC | **Databricks Apps** | Criar app de patient experience consumindo a view |
# MAGIC | **Agente** | Construir agente que monitora indicadores e envia briefing por e-mail |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Resumo do que você aprendeu
# MAGIC
# MAGIC - O que são AI Functions e por que democratizam IA
# MAGIC - Como usar `ai_classify`, `ai_analyze_sentiment`, `ai_extract`, `ai_summarize`, `ai_gen`, `ai_mask`
# MAGIC - Como combinar AI Functions com SQL tradicional para insights operacionais
# MAGIC - Como alimentar AI/BI Dashboards e Genie com colunas geradas por IA
# MAGIC - Boas práticas de custo, governança e limites da IA generativa em healthcare
# MAGIC
# MAGIC > **Databricks Field Engineering — Healthcare**
# MAGIC > Modern Data Intelligence Platform para saúde.
