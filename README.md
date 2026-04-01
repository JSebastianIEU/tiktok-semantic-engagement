# Semantic Similarity, Engagement Behavior, and Explainable Hashtag Recommendation on TikTok

**Course:** Natural Language Processing — IE University

**Authors:** Juan Sebastian Pena, Saad Ayomide

**Instructor:** Juan Jose Manjarín Colon

**Date:** 2026

---

## Data Access

> **Note:** The database credentials are **not included in this repository** because the data is private — collected from TikTok via a custom scraper stored in a private Supabase instance. The dataset is included directly in this repository as `tiktok_data.db` (SQLite). If you need to re-download from the source, use `download_data.py`.

### To regenerate the local data file

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
2. **Set your database credentials as environment variables** (contact the project authors for access):

   **Windows (PowerShell):**

   ```powershell
   $env:PG_HOST     = "aws-1-eu-west-1.pooler.supabase.com"
   $env:PG_PORT     = "5432"
   $env:PG_DBNAME   = "postgres"
   $env:PG_USER     = "postgres.mlmlcilyoqvbvgljsjtv"
   $env:PG_PASSWORD = "<your-password>"
   ```

   **macOS / Linux:**

   ```bash
   export PG_HOST="aws-1-eu-west-1.pooler.supabase.com"
   export PG_PORT="5432"
   export PG_DBNAME="postgres"
   export PG_USER="postgres.mlmlcilyoqvbvgljsjtv"
   export PG_PASSWORD="<your-password>"
   ```
3. **Run the download script** (one time only):

   ```bash
   python download_data.py
   ```

   This creates `tiktok_data.db` — a local SQLite copy of all 4 source tables from Supabase.
4. **Open the notebook in VSCode** with the Jupyter extension, or via:

   ```bash
   jupyter notebook NLP_Project_Notebook.ipynb
   ```

---

## Overview

TikTok's engagement landscape is famously unpredictable. A fitness tutorial and a cooking video can both go viral, but do similar kinds of content tend to generate *similarly patterned* engagement? This project explores that structural question — not asking *how much* engagement a post will receive, but whether **semantically similar captions form neighborhoods with more homogeneous engagement profiles** than randomly assembled groups of the same size.

The distinction matters. Predicting engagement levels from text alone is a task systematically defeated by non-textual factors: follower count, audio choice, thumbnail quality, posting time, and algorithmic amplification. But asking whether posts that *mean* similar things also *perform* similarly — that is a more tractable, and ultimately more interesting, question. The answer, across 22,647 TikTok posts and 12 independent statistical tests, is unambiguous: **yes, they do**.

**Research Question:** Do semantically similar TikTok captions (measured via transformer embeddings) form neighborhoods with lower within-group engagement variance than random groups of equal size? And can this structural regularity support an explainable hashtag recommendation system?

**H₀:** Semantic neighborhoods show no lower variance than random groups — proximity carries no engagement signal.
**H₁:** Semantic neighborhoods exhibit significantly lower within-group variance — posts that mean similar things perform more similarly than chance would predict.

**H₁ holds uniformly.** All 12 Mann-Whitney U tests across 4 engagement metrics and 3 neighborhood sizes returned p ≈ 0.0, with variance reductions of 10.5%–13.0% relative to random baselines.

---

## Data Source

The dataset was collected using a **custom TikTok scraper** built in a prior project and stored in a PostgreSQL database hosted on Supabase. The broader recommendation system framework is available at:

> https://github.com/PredictiveSocialMedia/Tik-Tok-Recommendation-System

**Database schema (normalised):**


| Table             | Key columns                                                            |
| ----------------- | ---------------------------------------------------------------------- |
| `videos`          | `video_id`, `caption`, `created_at`                                    |
| `video_snapshots` | `video_id`, `likes`, `comments_count`, `plays`, `shares`, `scraped_at` |
| `video_hashtags`  | `video_id`, `hashtag_id` (bridge table)                                |
| `hashtags`        | `hashtag_id`, `tag`                                                    |

**Dataset statistics (actual run):**

- 26,623 TikTok posts loaded (raw); **22,647 after cleaning** (3,976 dropped — captions with fewer than 4 non-hashtag characters)
- Fields used: `caption`, `hashtags` (aggregated via CTE), `likes`, `comments`, `views`, `shares`, `created_at`
- Hashtag coverage before pipeline fix: **3.9%** (883 / 22,647 posts — bridge table only)
- Hashtag coverage after combined extraction: **100%** (22,647 / 22,647; avg 5.58 hashtags/post, 36,866 unique tags)

The most striking number here is the coverage jump. Before implementing combined extraction, the bridge table covered fewer than 1 in 25 posts — a recommendation system built on that alone would be useless for over 95% of the corpus. The inline-caption extraction fix is not a preprocessing nicety; it is a prerequisite for the entire recommendation task.

---

## Project Structure

```
NLP_Project/
├── NLP_Project_Notebook.ipynb    # Main research notebook — all analysis and results
├── download_data.py              # One-time data download script (requires DB credentials)
├── tiktok_data.db                # Local SQLite copy of the TikTok dataset
├── README.md                     # This file — project documentation
└── requirements.txt              # Pinned Python dependencies
```

---

## Setup & Installation

### Prerequisites

- Python 3.10 or later
- VSCode with the Jupyter extension (recommended), or Jupyter Notebook/Lab

### Install dependencies

```bash
pip install -r requirements.txt
```

### First run — embeddings cache

SBERT embedding generation takes 5–15 minutes on CPU the first time. On subsequent runs the notebook loads from `embeddings_cache.npy` instantly. If you regenerate the dataset with new data, delete the cache first:

```bash
del embeddings_cache.npy       # Windows
rm  embeddings_cache.npy       # macOS / Linux
```

---

## How to Run

Open `NLP_Project_Notebook.ipynb` in VSCode (with the Jupyter extension) or via:

```bash
jupyter notebook NLP_Project_Notebook.ipynb
```

Then: **Kernel → Restart & Run All**

The notebook is self-contained: it reads from `tiktok_data.db`, generates embeddings (or loads from cache), runs all models and statistical tests, and produces the full set of visualizations and results.

---

## Key Results

### Classification Baselines

All four models converge to the same accuracy ceiling — which is itself the most important classification result.


| Model                   | Feature input                     | Accuracy | Macro F1 |
| ----------------------- | --------------------------------- | -------- | -------- |
| Logistic Regression     | SBERT`caption_clean` (384-dim)    | **59%**  | **0.59** |
| Multinomial Naive Bayes | TF-IDF`caption_tfidf` (demojised) | **60%**  | **0.58** |
| KNN k=5                 | SBERT`caption_clean` (384-dim)    | **59%**  | **0.59** |
| KNN k=10                | SBERT`caption_clean` (384-dim)    | **59%**  | **0.58** |

The convergence of dense (SBERT) and sparse (TF-IDF) representations at the same accuracy level is not a coincidence. It reflects an information ceiling: caption text, however richly encoded, simply does not contain the signals that drive TikTok engagement. This ceiling actually *strengthens* the structural finding — variance reduction cannot be explained by class separation, because no model can cleanly separate classes.

### Structural Analysis — Variance Reduction vs Random Baseline

All 12 Mann-Whitney U tests: **p = 0.0**. H₁ accepted across every metric and every neighborhood size.


| K  | likes_log | comments_log | views_log | shares_log |
| -- | --------- | ------------ | --------- | ---------- |
| 5  | 13.04%    | 12.72%       | 12.32%    | 12.46%     |
| 10 | 12.30%    | 12.74%       | 11.36%    | 11.98%     |
| 20 | 11.34%    | 11.27%       | 10.46%    | 10.53%     |

The monotonic decay from K=5 to K=20 is a structural signature: as neighborhood boundaries expand to include semantically more distant posts, within-group engagement coherence weakens and variance drifts toward the corpus-wide baseline. At K=5 — the tightest neighborhood — semantic coherence is strongest and variance reduction peaks at ~13%. This is the geometric property that makes KNN-based hashtag recommendation principled rather than arbitrary.

### Hashtag Coverage Improvement


| Method                                   | Posts with ≥1 hashtag                         |
| ---------------------------------------- | ---------------------------------------------- |
| Bridge table field only                  | 3.9% (883 / 22,647)                            |
| Combined (caption inline + bridge table) | **100%** (22,647 / 22,647; avg 5.58 tags/post) |

---

## Key Findings

The core finding of this project is **structural, not predictive**: semantically similar TikTok captions do not reliably predict engagement level (~60% accuracy, barely above the information ceiling), but they do attract more *similar* engagement patterns than random groups of the same size (10–13% variance reduction, p ≈ 0.0 across all conditions).

**Why variance reduction decreases with K:**
At K=5, only the five most semantically proximate posts form each neighborhood — these share tight topical and stylistic similarity, and their audiences substantially overlap, producing ~13% variance reduction. At K=20, the boundary expands to include more semantically distant posts, diluting the coherence and bringing variance closer to the random baseline (~11%). This monotonic decay is visible in the line plot (Section 7.6) and is a geometric consequence of the embedding space structure — it is precisely what a genuine semantic-engagement coupling should produce.

**Why both SBERT and TF-IDF achieve ~60% accuracy:**
Engagement is driven by factors absent from caption text. The richer semantic representations of SBERT offer no predictive advantage over keyword-frequency signals in TF-IDF at this ceiling. However, SBERT's geometric structure *is* more meaningful for the structural analysis: semantic distance in embedding space — not keyword overlap — is what captures the topical proximity that drives variance reduction.

**What the statistical tests confirm:**
Mann-Whitney U (alternative='less') tests whether semantic neighborhood variances are stochastically smaller than random group variances. p ≈ 0.0 across all 12 tests reflects both the strength of the effect and the scale of the experiment (22,647 neighborhoods per K value). The variance reduction percentages are the appropriate practical measure of effect size.

**The recommendation system is principled because the structural finding is real:**
The hashtag recommender does not guess. It surfaces tags that semantically neighboring posts have actually used, weighted by how frequently those tags appear and how much engagement they are associated with. The explainability is built-in: every recommendation can be traced back to specific real posts in the corpus that are semantically similar to the query.

---

## Pipeline Documentation


| Criterion                         | Notebook Section | Description                                                                                                                                                                 |
| --------------------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Data Collection & Cleaning** | Parts 1–2       | SQLite load from Supabase snapshot; combined hashtag extraction (3.9% → 100%); URL/mention/hashtag removal;`caption_clean` + `caption_tfidf` columns                       |
| **2. Preprocessing**              | Part 3           | Log-transformation y′=log(1+y); binary engagement label via median split on`views_log`; TF-IDF (10k features, bigrams, sublinear TF) fitted on demojised text              |
| **3. Feature Extraction**         | Part 4           | SBERT`all-MiniLM-L6-v2` (384-dim, L2-normalised) on `caption_clean`; UMAP 2D projection                                                                                     |
| **4. Modelling**                  | Parts 5–7       | FAISS/sklearn KNN for K∈{5,10,20}; Logistic Regression; Multinomial Naive Bayes; KNN classifier; semantic neighborhood structural analysis                                 |
| **5. Evaluation**                 | Parts 6–7       | Classification reports (macro F1, accuracy) for 4 models; confusion matrices; Mann-Whitney U; Levene's test; variance reduction heatmap; 1,000-simulation null distribution |
| **6. Deployment**                 | Part 9           | `recommend_for_new_caption()` — online inference for unseen captions; live demo (fitness, crypto, meal prep); productionisation guide (Streamlit / FastAPI / batch)        |
| **7. Code Documentation**         | This README      | Setup, pipeline map, results summary, usage instructions, dependency specification                                                                                          |

---

## Limitations


| Limitation                  | Impact                                                                                                                                                         |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Non-textual confounders** | Follower count, audio/music, thumbnail, posting time, and algorithmic amplification dominate engagement — none are present in caption text                    |
| **60% accuracy ceiling**    | An information limit, not a model limit; both dense (SBERT) and sparse (TF-IDF) representations converge here, confirming the bottleneck is in the data itself |
| **Multilingual content**    | `all-MiniLM-L6-v2` is primarily English-trained; non-English captions and TikTok-specific vernacular are embedded less reliably                                |
| **K sensitivity**           | Variance reduction ranges from ~10.5% (K=20, views) to ~13.0% (K=5, likes); practitioners must choose K based on precision vs recall preference                |
| **Selection bias**          | 3,976 posts (14.9%) dropped for having fewer than 4 clean characters — the working dataset skews toward posts with genuine descriptive content                |
| **Static corpus**           | KNN index is fixed; new posts require index updates; at scale, approximate nearest-neighbor structures (FAISS IVF) are needed for sub-second retrieval         |

---

## Methodology Notes

### Two-phase hashtag handling (critical design decision)

Hashtags must be **extracted before** cleaning and **removed from text before** SBERT encoding:

1. `extract_combined_hashtags(caption, hashtag_field)` — merges tags from both sources into `hashtag_list`
2. `build_caption_clean(text)` — removes hashtag strings, URLs, mentions; keeps raw emojis → `caption_clean`
3. `demojize_text(text)` — converts emojis to text tokens (e.g., 💪 → `flexed_biceps`) → `caption_tfidf`

Without step 2, SBERT would embed hashtag-string similarity rather than topical content, and nearest-neighbor retrieval would surface posts sharing the same tags rather than the same meaning. The hashtag is the label, not the feature.

### Emoji handling strategy


| Column          | Emoji treatment               | Used for                                                |
| --------------- | ----------------------------- | ------------------------------------------------------- |
| `caption_clean` | Raw Unicode emojis preserved  | SBERT (model handles Unicode natively)                  |
| `caption_tfidf` | Demojised via`emoji.demojize` | TF-IDF (emoji tokens become countable vocabulary items) |

---

## Dependencies

See `requirements.txt` for the full pinned list. Key packages:


| Package                          | Purpose                                                                 |
| -------------------------------- | ----------------------------------------------------------------------- |
| `sentence-transformers`          | SBERT embeddings (`all-MiniLM-L6-v2`)                                   |
| `scikit-learn`                   | TF-IDF, Logistic Regression, Naive Bayes, KNN, metrics                  |
| `umap-learn`                     | 2D embedding visualisation                                              |
| `scipy`                          | Mann-Whitney U test, Levene's test                                      |
| `emoji`                          | Emoji demojisation for TF-IDF input                                     |
| `sqlalchemy` + `psycopg2-binary` | SQLite connection (notebook); PostgreSQL connection (download_data.py)  |
| `faiss-cpu`                      | Fast cosine KNN for large corpora (optional; sklearn fallback included) |
| `pandas`, `numpy`                | Data manipulation                                                       |
| `matplotlib`, `seaborn`          | Visualisation                                                           |

---

## Reproducibility

- All stochastic operations seeded with `RANDOM_SEED = 42`
- Database credentials via environment variables only (never hardcoded)
- `embeddings_cache.npy` auto-invalidated on row-count mismatch
- `N_SIMULATIONS = 1000` for stable null distribution
