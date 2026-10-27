# Exercice 3 — Industrialiser et optimiser le RAG

Jusqu'ici, tout le code tenait dans des notebooks : pratique pour apprendre, mais impossible à réutiliser, à relancer à l'identique ou à brancher derrière une API. Dans cet exercice, vous allez :

1. transformer le RAG en une **vraie application Python**, découpée en modules ;
2. vous en servir pour **mesurer** l'effet de chaque optimisation, avec l'évaluation du chapitre 04 ;
3. **enregistrer** chaque expérience dans PostgreSQL, pour pouvoir les comparer.

## Le jeu de données : le Centre Hospitalier de Valmont

On travaille sur le corpus du chapitre 04 : les documents d'un hôpital fictif. **Tout est inventé** : patients, médecins, valeurs. Les données de santé sont très protégées par le RGPD ; générer des données synthétiques est une façon courante de développer un système sans exposer de vraies données. Les posologies sont plausibles, mais n'ont pas été vérifiées par un médecin.

### Trois fichiers

| Fichier | Contenu |
|---|---|
| `hopital/` | Les 280 documents |
| `hopital_documents.json` | Une fiche par document |
| `hopital_eval.json` | Le jeu d'évaluation (*golden dataset*) : 106 questions, avec la réponse attendue et la vérité terrain |

### Les documents

**247 documents de dossiers patients.** 60 patients fictifs, répartis en 10 parcours de soins de 6 patients chacun : diabète de type 2, insuffisance rénale chronique, fibrillation auriculaire, infarctus du myocarde, insuffisance cardiaque, BPCO, cholécystectomie, appendicite, prothèse totale de hanche, hypothyroïdie. Les documents s'étalent de janvier 2025 à juillet 2026.

| Type de document | Nombre | Format |
|---|---|---|
| Compte rendu de consultation (et passage aux urgences) | 103 | DOCX |
| Ordonnance | 36 | DOCX |
| Lettre de sortie d'hospitalisation | 36 | DOCX |
| Bilan biologique (tableaux de résultats) | 24 | DOCX |
| Compte rendu d'imagerie (échographie, scanner, radiographie) | 24 | PDF |
| Compte rendu opératoire | 18 | DOCX |
| Compte rendu de coronarographie | 6 | DOCX |

Les fichiers sont nommés `{IPP}_{date}_{type}`, par exemple `4100002_2025-04-13_ordonnance.docx`. L'IPP (identifiant permanent du patient) va de 4100001 à 4100060.

**33 notices de médicaments**, le livret thérapeutique de l'hôpital (`notice_metformine.docx`, par exemple). Elles couvrent les traitements des patients, les médicaments de l'anesthésie et le produit de contraste iodé. Toutes suivent les mêmes rubriques : classe thérapeutique, indications, posologie usuelle, contre-indications, précautions d'emploi (dont la conduite à tenir avant une chirurgie), interactions, effets indésirables.

### La fiche de chaque document

`hopital_documents.json` donne, pour chaque document, les metadata que l'on peut mettre dans le payload des chunks :

```json
{
  "document": "4100002_2025-04-13_ordonnance.docx",
  "patient_ipp": "4100002",
  "patient_name": "M. Bruno MARTIN",
  "care_path": "kidney",
  "doc_type": "ordonnance",
  "title": "Ordonnance",
  "date": "2025-04-13",
  "service": "Néphrologie",
  "format": "docx"
}
```

Pour une notice, `patient_ipp`, `patient_name` et `care_path` valent `null`.

### Le jeu d'évaluation

Chaque question de `hopital_eval.json` ressemble à ceci :

```json
{
  "id": "Q007",
  "type": "dose",
  "patient_ipp": "4100002",
  "question": "Quelle dose de ramipril prend actuellement M. Bruno Martin ?",
  "answer": "5 mg par jour.",
  "evidence": [
    {"document": "4100002_2025-04-13_ordonnance.docx", "snippets": ["Ramipril 5 mg"], "grade": 2, "fact": 0},
    {"document": "4100002_2025-07-19_consultation.docx", "snippets": ["ramipril 5 mg par jour"], "grade": 2, "fact": 0},
    {"document": "4100002_2025-04-13_consultation.docx", "snippets": ["à 5 mg par jour"], "grade": 1, "fact": null}
  ],
  "care_path": "kidney"
}
```

| Champ | Rôle |
|---|---|
| `type` | Le piège que la question teste (tableau ci-dessous) |
| `patient_ipp` | Le patient concerné, ou `null` (85 questions portent sur un patient précis) |
| `answer` | La réponse attendue : l'`expected_output` du LLM juge |
| `evidence` | La vérité terrain : les extraits de texte qui contiennent la réponse, document par document |
| `grade` | 2 : le passage contient la réponse, ou une partie nécessaire ; 1 : le passage est utile sans être nécessaire |
| `fact` | Le numéro du fait apporté. Même numéro : sources interchangeables ; numéros différents : faits complémentaires, tous nécessaires |

74 questions demandent un seul fait, 26 en demandent plusieurs (jusqu'à 5). Un chunk est pertinent s'il vient du bon document et contient tous les extraits d'une preuve : c'est la fonction `judge_chunk()` du chapitre 04. Cette vérité terrain ne dépend donc pas du chunking : elle reste valable pour toutes vos expériences.

| Type | Nombre | Ce que la question teste |
|---|---|---|
| `dose` | 8 | Une dose précise du dossier |
| `table` | 8 | Une valeur dans un tableau (bilan biologique, traitement de sortie) |
| `imaging` | 5 | Un compte rendu d'imagerie, en PDF |
| `abbreviation` | 8 | Une abréviation dans le document (« DT2 », « BPCO »), le mot complet dans la question |
| `lay_language` | 8 | Le langage courant dans la question (« crise cardiaque »), le terme médical dans le document |
| `buried` | 8 | Une information enfouie dans un long texte |
| `split` | 8 | Une information répartie sur deux paragraphes |
| `negation` | 8 | Une négation (« absence de perforation ») |
| `temporal` | 8 | La version la plus récente d'une information (dose modifiée au suivi) |
| `drug_info` | 5 | Une information générale d'une notice |
| `multi_hop` | 12 | Un fait du dossier **et** un fait d'une notice |
| `cross_patient` | 8 | Une liste de patients (« Quels patients sont allergiques à… ? ») |
| `cross_notice` | 6 | Une liste de médicaments du livret |
| `out_of_corpus` | 6 | Une question sans réponse dans le corpus : la bonne réponse est un refus |

Les 100 questions avec vérité terrain servent aux métriques déterministes (hit rate, MRR, recall, nDCG). Les réponses attendues servent au LLM juge (DeepEval). Les 6 questions `out_of_corpus` servent à tester les refus.

### Les pièges du corpus

Le corpus a été construit pour que les optimisations se voient dans les métriques. Les principaux pièges :

- **les patients aux profils proches** : six patients par parcours, avec les mêmes médicaments à des doses différentes ;
- **l'identité du patient seulement en en-tête** : une fois le document découpé, le chunk « Traitement de sortie » ne dit plus de quel patient il s'agit ;
- **les notices qui se ressemblent toutes** : 33 rubriques « Contre-indications » de même forme ;
- **la notice ou l'ordonnance** : la posologie usuelle d'un médicament (notice) ressemble à la posologie prescrite (ordonnance) ;
- **les questions de liste**, qu'un top 5 ne peut pas couvrir en entier.

## Les règles du jeu

- **Une configuration de référence d'abord** : celle du chapitre 04 (chunks de 512 tokens avec la ligne patient, bge-m3, recherche hybride, bge-reranker-v2-m3, 5 chunks). Toutes les expériences se comparent à elle.
- **Un seul changement à la fois.** Sinon, impossible de savoir ce qui a fait la différence.
- **Les métriques déterministes sur les 100 questions, pour chaque expérience** : elles ne coûtent presque rien.
- **Le LLM juge sur l'échantillon de 26 questions** (deux par type), seulement pour les changements qui touchent la génération et pour les configurations retenues.
- **Toute expérience est enregistrée** dans PostgreSQL : sa configuration complète et ses scores, question par question.

---

## 1. Sortir des notebooks

### Créer le projet

```bash
uv init --package rag-hopital
cd rag-hopital
```

L'option `--package` crée la structure `src/rag_hopital/`. Le projet est installé dans son propre environnement : les scripts, les notebooks et, plus tard, l'API peuvent tous faire `from rag_hopital.rag.pipeline import answer`.

### L'arborescence

```text
rag-hopital/
├── .env                          # URLs, keys, passwords: never committed
├── .env.example                  # the same keys without the values: committed
├── .gitignore
├── docker-compose.yml            # PostgreSQL + Qdrant
├── pyproject.toml
├── uv.lock
├── README.md                     # how to install and run everything
├── data/
│   ├── hopital/                  # the 280 documents
│   ├── hopital_documents.json
│   ├── hopital_eval.json         # the evaluation set
│   └── conversions/              # Docling JSON files, generated once
├── models/                       # embedders, rerankers, LLM: not committed
├── experiments/                  # one TOML file per experiment
│   ├── baseline.toml
│   └── chunks_256_overlap_32.toml
├── prompts/                      # system prompts, one text file each
│   ├── minimal.txt
│   └── strict.txt
├── scripts/                      # thin command-line entry points
│   ├── ingest.py                 # convert, chunk and index one configuration
│   ├── run_experiment.py         # retrieve, generate, evaluate, store the results
│   └── ask.py                    # ask one question from the terminal
├── notebooks/                    # exploration and analysis only
├── ui/                           # bonus: results dashboard
│   └── app.py
└── src/rag_hopital/
    ├── __init__.py
    ├── config.py                 # settings read from .env
    ├── db.py                     # PostgreSQL: connection, tables, inserts
    ├── etl/                      # documents -> chunks -> Qdrant
    │   ├── convert.py            # Docling
    │   ├── chunk.py              # size, overlap, patient line
    │   └── index.py              # embeddings, collections
    ├── rag/                      # question -> answer
    │   ├── search.py             # dense, sparse, hybrid
    │   ├── filters.py            # metadata filters (section 6)
    │   ├── rerank.py
    │   ├── generate.py           # prompt + LLM
    │   └── pipeline.py           # answer(question, config) -> answer + chunks
    ├── evaluation/
    │   ├── metrics.py            # hit rate, MRR, recall, nDCG
    │   ├── judge.py              # DeepEval metrics
    │   └── runner.py             # run a whole experiment
    └── api/                      # later: FastAPI, which only calls rag.pipeline
```

Trois dossiers, trois responsabilités : `etl/` prépare les données, `rag/` répond aux questions, `evaluation/` mesure. Le dossier `rag/` est le cœur de l'application : les scripts, l'évaluation et la future API l'appellent, sans jamais dupliquer sa logique.

Quelques règles :

- **Aucune logique dans `scripts/`, `notebooks/` ou `api/`** : ils lisent leurs arguments et appellent une fonction de `src/`.
- **Aucune URL, aucun mot de passe, aucun chemin en dur** dans le code.
- Ce qui dépend de **la machine** va dans `.env` ; ce qui dépend de **l'expérience** va dans un fichier TOML de `experiments/`.

### `.env` et `config.py`

```text
# .env.example
POSTGRES_URL=postgresql://user:password@localhost:5432/coursrag
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
EMBEDDINGS_URL=http://localhost:8001/v1
RERANK_URL=http://localhost:8002/rerank
LLM_URL=http://localhost:8000/v1
```

```python
# src/rag_hopital/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Connection settings, read from the .env file (or from environment variables)."""
    model_config = SettingsConfigDict(env_file=".env")

    postgres_url: str
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str
    embeddings_url: str = "http://localhost:8001/v1"
    rerank_url: str = "http://localhost:8002/rerank"
    llm_url: str = "http://localhost:8000/v1"


settings = Settings()
```

`pydantic-settings` lit le fichier `.env`, vérifie que les valeurs obligatoires sont présentes et convertit les types. Partout ailleurs : `from rag_hopital.config import settings`.

### Une expérience = un fichier TOML

```toml
# experiments/chunks_256_overlap_32.toml
name = "chunks_256_overlap_32"

[chunking]
max_tokens = 256
overlap = 32
patient_line = true

[embedder]
model = "bge-m3"
query_instruction = ""

[search]
method = "hybrid"
candidates = 30
top_k = 5

[reranker]
model = "bge-reranker"

[generation]
system_prompt = "prompts/strict.txt"
temperature = 0.0
top_p = 1.0
max_tokens = 512
```

Python lit le TOML sans bibliothèque supplémentaire, avec `tomllib`. Un script reste très court :

```python
"""Run one experiment: retrieval, generation, evaluation, results in PostgreSQL.

Usage: uv run python scripts/run_experiment.py experiments/baseline.toml
"""
import sys
import tomllib

from rag_hopital.evaluation.runner import run_experiment

with open(sys.argv[1], "rb") as f:
    config = tomllib.load(f)
run_experiment(config)
```

> **Une collection Qdrant par index.** Seuls le **chunking** et l'**embedder** obligent à réindexer. Nommez chaque collection d'après eux (par exemple `hopital__bge-m3__256_32`), et ne réindexez que si elle n'existe pas encore. Changer le reranker, le nombre de chunks ou la température ne demande aucune réindexation.

---

## 2. La taille des chunks et l'overlap

Testez plusieurs tailles de chunks (par exemple 128, 256 et 512 tokens) et plusieurs **overlaps** (0, puis environ 10 à 20 % de la taille).

L'**overlap** répète la fin d'un chunk au début du suivant. Une information à cheval sur deux chunks se retrouve ainsi entière dans au moins l'un des deux. Le principe, sur un découpage naïf en fenêtres de tokens :

```python
def sliding_window(text, tokenizer, size, overlap):
    """Naive chunking: windows of `size` tokens, each one starting `size - overlap` tokens after the previous one."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    step = size - overlap
    return [tokenizer.decode(ids[start:start + size]) for start in range(0, max(len(ids) - overlap, 1), step)]
```

Le `HybridChunker` de Docling n'a pas d'option d'overlap. On peut l'ajouter après coup, à la liste des chunks d'un même document :

```python
def add_overlap(texts, tokenizer, overlap):
    """Prefix each chunk of a document with the last `overlap` tokens of the previous chunk."""
    result = []
    for i, text in enumerate(texts):
        if i == 0 or overlap == 0:
            result.append(text)
        else:
            previous_ids = tokenizer.encode(texts[i - 1], add_special_tokens=False)
            result.append(tokenizer.decode(previous_ids[-overlap:]) + " " + text)
    return result
```

Points d'attention :

- appliquez l'overlap au **texte** des chunks (`chunk.text`), puis ajoutez la ligne patient et les titres ;
- un chunk avec overlap est plus long : donnez `max_tokens - overlap` au `HybridChunker` pour rester sous la limite de l'embedder ;
- regardez le **nombre de chunks** produit par chaque configuration : il pèse sur le temps d'indexation et la taille de la collection.

Les questions à surveiller : `split` (information sur deux paragraphes), `buried` (information enfouie dans un long texte), `table`.

---

## 3. Les paramètres du modèle

Deux familles de paramètres, à ne pas confondre.

**Les paramètres du retrieval** (métriques déterministes et LLM juge) :

- le nombre de chunks envoyés au LLM (3, 5, 10) ;
- le nombre de candidats transmis au reranker (10, 30, 50) ;
- la fusion de la recherche hybride : RRF (et son paramètre `k`) ou DBSF (`models.FusionQuery(fusion=models.Fusion.DBSF)`).

**Les paramètres de génération** (LLM juge uniquement : ils ne changent pas les chunks trouvés) :

- `temperature` : 0, 0,3, 0,7, 1 ;
- `top_p`, `top_k` : l'échantillonnage des tokens ;
- `max_tokens` : la longueur maximale de la réponse.

```python
response = llm_client.chat.completions.create(
    model=LLM_MODEL,
    messages=messages,
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    extra_body={"top_k": 40},      # llama-server parameter, not part of the OpenAI API
)
```

> Attention aux deux « top k » : le `top_k` de l'échantillonnage (le nombre de tokens candidats à chaque étape de la génération) n'a rien à voir avec le nombre de chunks envoyés au LLM.

Avec une température supérieure à 0, une même question donne des réponses différentes : lancez chaque configuration **trois fois** et regardez la dispersion des scores, pas seulement leur moyenne. Surveillez aussi les refus sur les 6 questions `out_of_corpus`.

---

## 4. Le prompt système

Le prompt système dit au LLM comment se servir des chunks. C'est souvent l'optimisation la moins chère, et l'une des plus efficaces. Testez au moins quatre prompts système, par exemple :

| Prompt | Ce qu'il ajoute | Ce qu'il devrait changer |
|---|---|---|
| `minimal.txt` | « Réponds à la question à partir des extraits fournis. », rien de plus | La référence basse : faithfulness, refus |
| `strict.txt` | Le prompt du chapitre 04 : uniquement les extraits, « Je ne sais pas. » sinon, une à trois phrases | Faithfulness, refus des questions `out_of_corpus` |
| `citations.txt` | `strict.txt` + citer le numéro de l'extrait après chaque information, par exemple [2] | Faithfulness, réponses vérifiables |
| `patient.txt` | `strict.txt` + vérifier que chaque extrait utilisé concerne bien le patient nommé dans la question | Right patient, correctness |

D'autres variantes possibles : un rôle détaillé ou aucun rôle, une réponse courte ou détaillée, un prompt rédigé en anglais (avec des réponses toujours en français), les chunks dans le prompt système plutôt que dans le message de l'utilisateur, l'ordre des chunks (le meilleur en premier ou en dernier).

Un prompt est un fichier texte, versionné comme le code, et désigné dans la configuration de l'expérience :

```python
from pathlib import Path

system_prompt = Path(config["generation"]["system_prompt"]).read_text(encoding="utf-8")
```

Points d'attention :

- **Enregistrez le texte du prompt** dans la configuration de l'expérience en base, pas seulement le nom du fichier : un fichier peut être modifié après coup, et l'expérience ne serait plus reproductible.
- **Le retrieval ne change pas** d'un prompt à l'autre : les métriques déterministes restent identiques. Seul le LLM juge mesure la différence. Gardez une température de 0.
- **Mesurez les refus dans les deux sens** : un prompt qui refuse plus souvent les questions `out_of_corpus` refuse peut-être aussi des questions qui ont une réponse. Comptez les deux :

```python
def is_refusal(answer):
    """The system prompt asks for this exact sentence when the passages do not answer the question."""
    return "je ne sais pas" in answer.lower()
```

- **Vérifiez les citations** de `citations.txt` sans LLM : les numéros cités existent-ils, et les chunks cités sont-ils pertinents d'après la vérité terrain (`judge_chunk()`) ?

```python
import re

def cited_passages(answer):
    """Numbers of the passages cited in the answer, written like [2]."""
    return {int(number) for number in re.findall(r"\[(\d+)\]", answer)}
```

---

## 5. D'autres embedders et rerankers

| Rôle | Référence | À comparer |
|---|---|---|
| Embedder | bge-m3 | Harrier-OSS-v1-0.6b (Microsoft), Qwen3-Embedding-0.6B |
| Reranker | bge-reranker-v2-m3 | Qwen3-Reranker-0.6B |

Points d'attention :

- **Lisez la model card.** Harrier et Qwen3-Embedding attendent une **instruction** devant les questions, au format `Instruct: {tâche}\nQuery: {question}`, et rien devant les documents. bge-m3 n'en veut pas. Sans instruction, les scores baissent. Mettez-la dans la configuration :

```python
def query_text(question, config):
    """Harrier and Qwen3-Embedding expect a task instruction before the question; documents get none."""
    instruction = config["embedder"]["query_instruction"]
    return f"Instruct: {instruction}\nQuery: {question}" if instruction else question
```

- **Une collection par embedder** : les vecteurs de deux modèles différents ne sont pas comparables. Lisez la taille des vecteurs dans la configuration du modèle, ne la supposez pas.
- **Qwen3-Reranker ne se sert pas tel quel avec vLLM.** Il faut des options supplémentaires et un template de prompt fourni par vLLM :

```bash
vllm serve models/reranker/qwen3-reranker-0.6b --runner pooling --served-model-name qwen3-reranker --port 8002 \
    --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}' \
    --chat-template models/reranker/qwen3_reranker.jinja
```

- **Mesurez aussi le coût** : temps d'indexation, temps de réponse par question, mémoire GPU (`nvidia-smi`). Un modèle un peu meilleur mais trois fois plus lent n'est pas forcément le bon choix.

Les questions à surveiller : `abbreviation` et `lay_language` pour les embedders, `negation` et le MRR pour les rerankers.

---

## 6. Extraire des filtres de la question avec un LLM

Six patients sont sous metformine ; quand la question nomme le patient, pourquoi chercher dans les dossiers des cinq autres ? L'idée : un LLM lit la question et en extrait des **filtres** (patient, type de document, période), qu'on applique à la recherche dans Qdrant. C'est le principe du *self-querying retriever*.

C'est une bonne idée **à condition de la contrôler** :

- **le LLM extrait, le code décide.** Demandez au LLM un JSON (`{"patient_name": ..., "document_type": ...}`), puis retrouvez l'IPP dans la liste des patients connus (`hopital_documents.json`). Ne faites jamais confiance à un identifiant écrit par le LLM ;
- **un mauvais filtre est pire que pas de filtre** : s'il exclut le bon patient, le recall tombe à 0. Prévoyez un repli : si la recherche filtrée ne renvoie rien, relancez-la sans filtre ;
- **ne filtrez que ce qui est sûr** : une question sans patient (`drug_info`, `cross_notice`, `cross_patient`) ne reçoit pas de filtre patient.

Le filtre dans Qdrant (les champs filtrés doivent avoir un *payload index*) :

```python
patient_filter = models.Filter(must=[
    models.FieldCondition(key="patient_ipp", match=models.MatchValue(value=ipp)),
])
points = qdrant.query_points(
    collection,
    prefetch=[
        models.Prefetch(query=dense_vector, using="dense", limit=candidates),
        models.Prefetch(query=bm25(question), using="sparse", limit=candidates),
    ],
    query=models.RrfQuery(rrf=models.Rrf(k=60)),
    query_filter=patient_filter,          # also applied to the prefetches
    limit=top_k,
).points
```

Pour filtrer sur le type de document ou la date, ajoutez d'abord ces champs au payload des chunks : ils sont dans `hopital_documents.json`.

À mesurer :

1. **La qualité de l'extraction elle-même** : le jeu d'évaluation donne le `patient_ipp` de chaque question. Comptez les IPP correctement extraits, les erreurs et les oublis.
2. **L'effet sur la recherche** : avec et sans filtre, par type de question.
3. **Une solution sans LLM** : les 60 noms de patients sont connus. Une simple recherche de ces noms dans la question fait-elle aussi bien, sans appel au LLM ni latence supplémentaire ?

Les questions à surveiller : `dose`, `table`, `temporal`, et la confusion entre la notice d'un médicament et l'ordonnance d'un patient.

---

## 7. Tout évaluer, tout enregistrer

Chaque exécution de `run_experiment.py` enregistre sa configuration, les réponses et les scores dans PostgreSQL :

```sql
CREATE TABLE experiments (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    config      JSONB NOT NULL,                  -- the whole TOML file
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    duration_s  REAL,
    status      TEXT NOT NULL DEFAULT 'running'  -- running, done, failed
);

CREATE TABLE answers (
    experiment_id  INTEGER NOT NULL REFERENCES experiments(id),
    question_id    TEXT NOT NULL,
    chunks         JSONB NOT NULL,               -- document, chunk index and score of each retrieved chunk
    answer         TEXT,
    latency_ms     REAL,
    PRIMARY KEY (experiment_id, question_id)
);

CREATE TABLE scores (
    experiment_id  INTEGER NOT NULL REFERENCES experiments(id),
    question_id    TEXT NOT NULL,
    question_type  TEXT NOT NULL,
    metric         TEXT NOT NULL,                -- hit@5, mrr, recall@5, ndcg@5, faithfulness...
    value          REAL,                         -- NULL when the judge failed
    reason         TEXT,                         -- the judge's explanation, if any
    PRIMARY KEY (experiment_id, question_id, metric)
);
```

Une ligne par score, et non une colonne par métrique : on ajoute une métrique sans toucher au schéma, et les métriques déterministes et celles du juge vivent dans la même table.

Comparer toutes les expériences devient une requête SQL :

```sql
SELECT e.name, s.metric, round(avg(s.value)::numeric, 3) AS mean, count(s.value) AS n
FROM scores s
JOIN experiments e ON e.id = s.experiment_id
GROUP BY e.name, s.metric
ORDER BY s.metric, mean DESC;
```

Et la configuration reste interrogeable, grâce au JSONB : `e.config -> 'chunking' ->> 'overlap'`.

---

## 8. D'autres pistes d'optimisation

Choisissez-en au moins deux :

| Piste | Idée | Questions à surveiller |
|---|---|---|
| Ligne patient | Retirer la ligne patient en tête des chunks, pour mesurer ce qu'elle apporte | Toutes celles qui portent sur un patient |
| Contextual retrieval | Un LLM rédige une phrase de contexte pour chaque chunk avant l'embedding | `buried`, `split` |
| Parent-child | Chercher avec de petits chunks, envoyer au LLM la section entière qui les contient | `split`, `buried` |
| Décomposer la question | Un LLM découpe une question en sous-questions (le dossier, puis la notice), une recherche par sous-question | `multi_hop` |
| Réécrire la question | Plusieurs reformulations, ou une réponse hypothétique cherchée à la place de la question (HyDE) | `lay_language`, `abbreviation` |
| Tableaux | Une autre représentation des tableaux dans les chunks (une ligne par chunk, phrases) | `table` |
| Seuil du reranker | Répondre « Je ne sais pas » si le meilleur score du reranker est trop bas | `out_of_corpus` |
| Quantification | Vecteurs compressés dans Qdrant (scalaire, binaire) : la qualité baisse-t-elle ? | Toutes |
| Un autre LLM | Comparer deux LLM pour la génération ; ou juger les mêmes réponses avec un second juge, d'une autre famille, pour mesurer le biais du juge qui note ses propres réponses | Métriques du juge |

---

## Bonus — Une interface pour analyser les résultats

Développez, avec l'aide d'un assistant de code, une petite interface (Streamlit, par exemple) qui lit la base PostgreSQL :

- un **classement** des expériences, métrique par métrique ;
- la **comparaison de deux expériences** : par type de question, et les questions améliorées ou dégradées ;
- le **détail d'une question** : la réponse attendue, les chunks trouvés et la réponse de chaque expérience, les explications du juge.

Règles : l'interface ne fait que **lire** la base ; elle importe ce dont elle a besoin depuis `src/` au lieu de le réécrire ; et vous devez pouvoir expliquer chaque ligne du code généré.

---

## Livrables

- Le dépôt, avec l'arborescence de la section 1 et un `README.md` qui explique comment tout installer et lancer.
- La base PostgreSQL remplie : la référence et au moins une expérience par section 2 à 6, plus deux pistes de la section 8.
- Un court `RAPPORT.md` :
  - un tableau comparatif des expériences ;
  - la meilleure configuration, et pourquoi (qualité, mais aussi coût et temps de réponse) ;
  - ce qui n'a **rien** changé, ou a dégradé les résultats ;
  - trois questions ratées par la meilleure configuration, analysées : où se situe l'échec, et quelle piste pourrait le corriger.
