# Hands-On HSL — Lakeflow Designer + AI Functions

> Workshop técnico-executivo Databricks focado em saúde — do setup do Lakehouse à IA Generativa aplicada ao contexto hospitalar.

---

## Conteúdo

| Arquivo | O que faz | Quando usar |
|---------|-----------|-------------|
| **`Healthcare_Lakehouse_Demo_v1.py`** | Setup completo do Lakehouse (catálogo `healthcare_lakehouse` fixo) + cadastros base + landing zone | Demo individual, single-tenant |
| **`Healthcare_AI_Functions_v1.py`** | AI Functions sobre os dados criados na v1 | Demo individual, single-tenant |
| **`Healthcare_Lakehouse_Demo_v2.py`** | Mesmo setup, catálogo **parametrizado** via widget `healthcare_lakehouse_<sufixo>` | Workshops com múltiplos participantes |
| **`Healthcare_AI_Functions_v2.py`** | AI Functions consumindo o catálogo parametrizado | Workshops com múltiplos participantes |
| **`databricks-academy.png`** | Logo do cabeçalho dos notebooks | — |

---

## v1 vs v2 — qual usar?

| | v1 | v2 |
|---|----|----|
| **Catálogo** | `healthcare_lakehouse` (fixo) | `healthcare_lakehouse_<sufixo>` (parametrizado) |
| **Input** | nenhum | widget `sufixo` no topo do notebook |
| **Caso de uso** | individual | workshops, equipes, multi-tenant |
| **Exemplo** | `healthcare_lakehouse.bronze.hospitais` | `healthcare_lakehouse_leandro.bronze.hospitais` |

Tudo o mais (esquema, dados sintéticos, AI Functions, dashboards) é **idêntico** entre as duas versões.

---

## AI Functions demonstradas

| Função | O que faz |
|--------|-----------|
| `ai_classify(text, ARRAY)` | Classifica texto em categorias (prioridade clínica, nível de risco) |
| `ai_analyze_sentiment(text)` | Detecta sentimento em avaliações de pacientes |
| `ai_extract(text, ARRAY)` | Extrai campos estruturados de texto livre (sintomas, duração) |
| `ai_summarize(text, max_words)` | Resume textos longos |
| `ai_gen(prompt)` | Geração livre — recomendações, insights executivos |
| `ai_mask(text, ARRAY)` | Anonimização de PII/PHI (LGPD/HIPAA) |

---

## Como começar

### Opção 1 — Git folder no Databricks (recomendado)

1. No workspace Databricks, abra sua pasta de usuário (**Workspace → Users → seu_email**)
2. Clique em **"+" → Git folder**
3. Cole: `https://github.com/Databricks-BR/hands-on-hsl.git`
4. **Create Git folder**
5. Abra o notebook desejado (`_v1` ou `_v2`) → **Run all**

### Opção 2 — Download ZIP

**Code → Download ZIP** → importar no Databricks via **Workspace → Import**

---

## Como funciona a v2 (parametrização)

Os notebooks v2 começam com 2 células de setup:

```sql
-- Célula 1: cria widget no topo do notebook
CREATE WIDGET TEXT sufixo DEFAULT "leandro";

-- Célula 2: monta o nome do catálogo dinamicamente
DECLARE OR REPLACE VARIABLE catalog_name STRING;
SET VAR catalog_name = CONCAT('healthcare_lakehouse_', :sufixo);
```

Daí em diante:
- DDL: `CREATE CATALOG IDENTIFIER(catalog_name)`
- Queries: `healthcare_lakehouse_${sufixo}.bronze.hospitais` (substituição textual)
- Volumes: `/Volumes/healthcare_lakehouse_${sufixo}/bronze/landing_zone/...`

| Sufixo digitado | Catálogo criado |
|-----------------|------------------|
| `leandro` (default) | `healthcare_lakehouse_leandro` |
| `workshop01` | `healthcare_lakehouse_workshop01` |
| `equipe_a` | `healthcare_lakehouse_equipe_a` |

> **Regra:** apenas letras minúsculas, números e underscore. Sem hífen, espaço ou acento.

---

## Pré-requisitos

| Item | Requisito |
|------|-----------|
| **Workspace** | Databricks com Unity Catalog habilitado |
| **Compute** | SQL Warehouse Serverless **ou** Serverless Compute |
| **Permissões** | `CREATE CATALOG` no metastore |
| **Runtime** | DBR 13+ |

Os notebooks são **100% SQL** — sem pip install, sem cluster custom.

---

## Roadmap completo do hands-on

| Módulo | Onde acontece | Conteúdo |
|--------|--------------|----------|
| **1 — Setup** | Notebook SQL (`Healthcare_Lakehouse_Demo_*`) | Catálogo, schemas, cadastros base, landing zone |
| 2 — Ingestão de consultas | **Lakeflow Designer** | Pipeline visual: CSV → Bronze → Silver |
| 3 — Ingestão de avaliações | **Lakeflow Designer** | Pipeline visual: CSV → Bronze |
| 4 — Enriquecimento (Visão 360) | **Lakeflow Designer** | Joins visuais entre fato e dimensões |
| **5 — AI Functions** | Notebook SQL (`Healthcare_AI_Functions_*`) | `ai_classify`, `ai_analyze_sentiment`, etc |
| 6 — Gold & Dashboards | **AI/BI Dashboards** | KPIs e visualizações executivas |
| 7 — Apps & Genie | **Databricks Apps + Genie Spaces** | App de patient experience + Q&A em linguagem natural |

---

## Licença

Material de treinamento Databricks Brasil. Uso livre para fins educacionais.

---

**Databricks Field Engineering — Healthcare**
