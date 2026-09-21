# Data Vortex - AARUUSH'26

**Theme:** Rebuilding the Social Engine  
**Event:** AARUUSH'26, SRM Institute of Science and Technology  
**Dates:** September 13-23, 2026  
**Team:** Entropy  
**Members:** Saanvi Grover & Aditya Sharma

---

## Competition Overview

Data Vortex is a multi-round data science competition centered around recovering and analyzing a corrupted social media platform dataset ("The Social Engine"). Round 1 covers the recovery of the dataset and its analytical foundation in two phases: data cleaning with exploratory analysis, followed by SQL-based analytical reasoning. Round 2, "Semantic Recovery," moves from structured recovery to the platform's comprehension layer: recovering the sentiment and topic carried by its posts using NLP. Round 3, "Signal Tracking," takes the restored model live: collecting real public reaction to an assigned topic from public APIs and detecting behavioural shifts as they happen. Round 4, "Social Engine Revival," brings every restored component together in an interactive dashboard.

## Dataset

### Round 1 — The Social Engine dataset
Retrieved from a simulated corrupted website (`datavortex-social-engine.vercel.app`) through an interactive recovery terminal. Consists of:

- **Posts:** 12,000 social media posts with text content, engagement metrics (likes, shares, comments), timestamps, and platform tags
- **Users:** 1,500 user profiles with location, language, account creation date, and follower count
- **Platforms:** Facebook, YouTube, Twitter, Reddit, Instagram
- **Date Range:** May 2024 - April 2025

### Round 2 — Dataset 2
9,000 labelled social-media posts, separate from the Round 1 corpus, with two targets: `sentiment_label` (Negative / Neutral / Positive, perfectly balanced) and `topic_category` (4 classes, 86% one class). Only 7,900 of the 9,000 rows are unique text — see [Round 2 Key Findings](#round-2-semantic-recovery-1) below.

### Round 3 — self-collected live data
Public reaction to the **iOS 27 / iPadOS 27 / macOS 27** release (14 September 2026), with **Android 17** as a comparison, collected from Reddit, Hacker News, Mastodon and Lemmy with no paid APIs: **5,498 posts and comments** over a 297.6-hour collection window in two snapshots, of which **1,403** (1,132 distinct authors) explicitly mention a tracked release. Author names are stored only as salted hashes.

## Project Structure

```
Data Vortex/
|
|-- Round1-Phase1-Data-Recovery/      # Data cleaning + EDA
|   |-- Entropy_01_Data_Cleaning.ipynb          # 10-step cleaning pipeline with before/after examples
|   |-- Entropy_02_EDA_Report.ipynb             # 20-section EDA, 21 visualizations, 6 statistical tests
|   |-- Entropy_Phase1_EDA_Report.pdf           # Standalone EDA insight report (LaTeX)
|   |-- Entropy_Social_Engine_Cleaned.csv       # Single merged submission file (12,000 rows x 12 cols)
|   |-- Entropy_Social_Engine_Posts_Cleaned.csv # Cleaned posts (12,000 rows)
|   |-- Entropy_Social_Engine_Users_Cleaned.csv # Cleaned users (1,500 rows)
|   |-- Entropy_Social_Engine_Posts_Corrupted.csv
|   |-- Entropy_Social_Engine_Users.csv
|   |-- Entropy_cleaning_log.txt                # Log of every transformation applied
|   |-- images/                                 # 21 EDA chart PNGs
|   |-- Entropy_README_Round1_Phase1.md
|
|-- Round1-Phase2-Analytical-Core/    # SQL analysis (E3 + M4 + H4)
|   |-- Entropy_Phase2_SQL_Queries.pdf          # Deliverable 1: final SQL queries
|   |-- images/Entropy_{E3,M4,H4}_output.jpeg   # Deliverable 2: output screenshots
|   |-- Entropy_Phase2_Logic_Explanation.pdf    # Deliverable 3: logic and approach
|   |-- Entropy_Phase2_Insight_Report.pdf       # Deliverable 4: insights and findings
|   |-- Entropy_01_SQL_Analysis.ipynb           # Builds constrained schema, runs the 3 queries, stat checks (executed)
|   |-- Entropy_social_engine.db                # SQLite database (PK/FK, CHECK constraints, indexes, view)
|   |-- queries/                                # The 3 final .sql files
|   |-- images/                                 # Output screenshots + 4 insight charts
|   |-- Entropy_README_Round1_Phase2.md
|
|-- Round2-Semantic-Recovery/          # NLP: sentiment + topic recovery
|   |-- code/
|   |   |-- Entropy_01_NLP_Model.ipynb           # Main end-to-end notebook: EDA -> leakage check -> preprocessing -> model selection -> evaluation -> error analysis
|   |   |-- Entropy_nlp_utils.py                 # NLP utilities: text cleaning, NB-SVM, topic rules, sentiment ensemble loader
|   |
|   |-- data/
|   |   |-- Entropy_Labeled_Social_NLP_Training_Data.xlsx # Main labeled dataset used for NLP model development and evaluation
|   |
|   |-- model/
|   |   |-- Entropy_minilm_sentiment_seed7/       # MiniLM sentiment model trained with seed 7
|   |   |-- Entropy_minilm_sentiment_seed42/      # MiniLM sentiment model trained with seed 42
|   |   |-- Entropy_minilm_sentiment_seed1234/    # MiniLM sentiment model trained with seed 1234
|   |   |-- Entropy_r2_model_artifacts.zip        # Compressed archive containing Round 2 model artefacts
|   |   |-- Entropy_sentiment_embedding_logreg.pkl # Logistic Regression model using text embeddings for sentiment
|   |   |-- Entropy_sentiment_model.json           # Sentiment model configuration and metadata
|   |   |-- Entropy_sentiment_tfidf_logreg.pkl     # TF-IDF + Logistic Regression sentiment classifier
|   |   |-- Entropy_topic_char_logreg.pkl          # Character n-gram TF-IDF + Logistic Regression topic classifier
|   |   |-- Entropy_topic_rules.json               # Rule-based topic classification patterns and triggers
|   |
|   |-- outputs/
|   |   |-- Entropy_r2_classification_report.xlsx # Detailed classification metrics for the final model
|   |   |-- Entropy_r2_metrics.json                # Exported evaluation metrics in JSON format
|   |   |-- Entropy_r2_model_comparison.xlsx       # Performance comparison of evaluated models
|   |   |-- Entropy_r2_test_predictions.xlsx       # Test-set predictions with actual and predicted labels
|   |
|   |-- reports/
|   |   |-- images/                              # 8 report figures used for analysis and reporting
|   |   |   |-- Entropy_r2_class_distribution.png  # Class/sentiment distribution visualisation
|   |   |   |-- Entropy_r2_confusion_matrix.png    # Confusion matrix for classification performance
|   |   |   |-- Entropy_r2_error_slices.png        # Model error analysis across different data slices
|   |   |   |-- Entropy_r2_learning_curve.png      # Learning curve showing performance vs training data
|   |   |   |-- Entropy_r2_model_comparison.png    # Visual comparison of candidate model performance
|   |   |   |-- Entropy_r2_topic_confusion_word_model.png # Topic classification word/confusion analysis
|   |   |   |-- Entropy_r2_topic_substring_triggers.png # Topic substring trigger analysis
|   |   |   |-- Entropy_r2_training_curves.png     # Training and validation curves
|   |   |
|   |   |-- Entropy_r2_evaluation_metrics_report.pdf # Standalone metrics + error-analysis report
|   |   |-- Entropy_r2_evaluation_metrics_report.tex # LaTeX source for the evaluation metrics report
|   |   |-- Entropy_r2_technical_report.pdf        # Complete technical report covering methodology and results
|   |   |-- Entropy_r2_technical_report.tex        # LaTeX source for the technical report
|   |
|   |-- Entropy_README_Round2.md
|
|-- Round3-Signal-Tracking/           # Live collection + real-time analysis
|   |-- Entropy_collect_data.py                 # Deliverable 2: key-less collector (Reddit/HN/Mastodon/Lemmy/news)
|   |-- Entropy_02_Realtime_Analysis.ipynb      # Deliverable 3: real-time analysis notebook (executed)
|   |-- Entropy_03_robustness_checks.py         # CIs, author concentration, VADER baseline, alert replay, kappa
|   |-- data/
|   |   |-- Entropy_round3_collected_posts.csv  # Deliverable 1: 5,498 collected posts and comments
|   |   |-- Entropy_round3_news_timeline.csv    # 390 news articles used to date real events
|   |   |-- Entropy_round3_collection_manifest.json
|   |   |-- raw/request_log.json.gz             # every HTTP request with status
|   |-- outputs/
|   |   |-- Entropy_round3_scored_posts.csv     # 1,403 on-topic posts with model sentiment
|   |   |-- Entropy_round3_metrics.json         # every number quoted in the report
|   |   |-- Entropy_round3_robustness.json      # period CIs, author concentration, VADER, alert replay
|   |   |-- Entropy_round3_annotation_{saanvi,aditya}.csv  # two independent annotators, 150 posts each
|   |   |-- Entropy_round3_consensus_labels.csv # the 140 they agreed on - the reference standard
|   |-- images/                                 # 9 analysis figures
|   |-- reports/
|   |   |-- Entropy_round3_analytical_report.pdf  # Deliverable 4 (+ .tex source)
|   |   |-- Entropy_r3_analysis_notebook.pdf      # notebook exported to PDF for submission
|   |   |-- Entropy_r3_common.tex                 # shared LaTeX preamble
|   |-- Entropy_README_Round3.md
|
|-- Round4-Social-Engine-Revival/     # Interactive dashboard integrating all rounds
|   |-- Entropy_app.py                          # Streamlit + Plotly dashboard (5 pages, live artefact reads)
|   |-- Entropy_post_fetcher.py                 # resolves a public post URL to its text
|   |-- Entropy_rag.py                          # retrieval index + Groq call behind SHANNON
|   |-- Entropy_requirements.txt                # full set, including the optional model extras
|   |-- .streamlit/config.toml                  # pinned deep-forest theme
|   |-- .env.example                            # copy to .env and add GROQ_API_KEY
|   |-- static/Entropy_bg.webp                  # background artwork (served at app/static/)
|   |-- Entropy_README_Round4.md
|
|-- requirements.txt                  # lean set Streamlit Community Cloud installs
|-- .streamlit/config.toml            # same theme, at the root so Cloud picks it up
|-- README.md                         # This file
```

## Progress

### Round 1

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 - Data Recovery | Completed | Data cleaning (10 justified transformations) + EDA (20 sections, 21 visualizations, 6 statistical tests) + PDF insight report |
| Phase 2 - Analytical Core | Completed | One question per level (E3, M4, H4) with SQL query PDF, JPEG output screenshots, logic explanation PDF, and insight report PDF |

### Round 2

| Round | Status | Description |
|-------|--------|-------------|
| Round 2 - Semantic Recovery | Completed | Leakage-checked 70/15/15 split, preprocessing ablation, 8-model comparison, weighted 3-component ensemble (TF-IDF+LogReg, MiniLM embeddings+LogReg, 3-seed fine-tuned MiniLM), full error analysis, spurious topic-label rule recovery, Technical Report PDF, Evaluation Metrics Report PDF |

### Round 3

| Round | Status | Description |
|-------|--------|-------------|
| Round 3 - Signal Tracking | Completed | Key-less collector for 4 public platforms (Reddit Atom, HN Algolia, Mastodon, Lemmy) + news timeline, two snapshots, Round 2 model applied unchanged and validated in-domain on 150 hand-labelled posts, 3 significant sentiment shifts, 44 spike hours, entity/theme/NMF topic analysis, trigger table evidenced by news and named posts, Analytical Report PDF |

### Round 4

| Round | Status | Description |
|-------|--------|-------------|
| Round 4 - Social Engine Revival | Completed | Deep-forest Streamlit + Plotly dashboard, ~34 charts across 5 pages, navigated by a pinned clickable flowchart of the four-round pipeline whose last node is the dashboard itself; KPI tiles on every page and a vertical methodology flowchart with a live artefact panel after each round's analysis, a platform filter, a time-window slider, the Round 1b SQL run live against the database, the Round 2 ensemble loaded for on-demand inference, the Round 3 live corpus re-binned, a live operating-point slider that re-decides all 1,403 posts from saved probabilities, bootstrap intervals on the sentiment timeline, a filterable evidence table of the scored corpus, four cross-round journey charts (scale per round, every model built against its live score, a Lorenz curve of synthetic vs live engagement, training balance vs the live world), every chart paired with an interpretation and a "why it matters" note, each round's submitted report PDFs plus its executed notebook (code, outputs and figures) rendered at the end of its own page, a live predictor that scores typed text or any public post URL, and **SHANNON**, a retrieval-augmented assistant answering questions from the repository itself |

## Key Findings

### Phase 1: Data Recovery
- Identified and fixed **10 corruption patterns** including NULL strings, duplicates, HTML entities, mojibake, inconsistent timestamps, and negative values
- Removed **360 duplicate rows**, corrected **509 negative likes**, decoded **974 HTML artifacts**, and recovered **306 mojibake instances** to their original Unicode characters
- ~15% missing data across platform, text, and likes columns, confirmed **Missing Completely at Random** (Mann-Whitney U, p = 0.64)
- The dataset is **structurally uniform**: platform volume, day-of-week patterns, brand mentions, and language distribution all show coefficient of variation below 10%
- **5 of 6 statistical tests** return p > 0.05; the one significant signal is that **multi-platform users** show ~15% higher engagement
- Contradictory sentiment phrases in the same post reveal **template-based text generation**

### Phase 2: Analytical Core

| Level | Question | Answer | What it means |
|-------|----------|--------|---------------|
| Easy | E3 - Average Engagement by Platform | Instagram, 4,040.0 avg total engagement | All platforms within **2.2%**; the data could detect a 3.7% gap and finds none (Kruskal-Wallis p = 0.56) |
| Medium | M4 - Platform Behaviour by High-Follower Users | Instagram, 4,145.2 for the 594 users with 30k+ followers | **+4.4%** over smaller accounts, but not significant after Holm correction (p = 0.12), and the platform ranking reverses between groups |
| Hard | H4 - Follower-to-Engagement Anomaly | 16 users with < 5k followers in the top 10% | Close to the **13.1** expected by chance (p = 0.38); 14 are volume-driven and **2 are exceptional per post** (`user_ogtvuuki`, `user_kbdvf8d6`) |

- Two-table normalized schema with engine-enforced primary/foreign keys, NOT NULL and CHECK constraints; `EXPLAIN QUERY PLAN` confirms the indexes drive each query
- Every query carries its own benchmark: deviation from the overall mean (E3), comparison with users under 30k (M4), and a per-post decile ranking (H4); rankings are deterministic
- Corrupted `likes` handled per question: SQL NULL semantics for E3 averages, the brief's engagement definition for M4, per-user imputation for H4 totals (zero-filling would change 2 of 16 names)
- **Posting volume, not followers, drives user totals:** post count vs total engagement Spearman rho = 0.91, followers vs total rho = -0.01

### Round 2: Semantic Recovery

**Selected model (sentiment):** a weighted ensemble — `p̂ = 0.25·p(TF-IDF+char LogReg) + 0.20·p(MiniLM embeddings+LogReg) + 0.55·p(fine-tuned MiniLM)` — chosen by grid search on validation, before the test set was touched.

| Metric | Value |
|---|---|
| Test macro-F1 | **0.7178** (95% CI [0.6913, 0.7414]) |
| Test accuracy | 0.7173 |
| Test ROC-AUC (OvR) | 0.8805 |
| vs. best classical baseline | +0.0935 macro-F1 (McNemar p = 3.0×10⁻¹⁰) |
| Topic labels (recovered substring rules) | 1.0000 accuracy / macro-F1 |

- **Deduplicate before splitting, or the score is a lie:** only 7,900 of 9,000 rows are unique text; a naive random split leaks duplicates across train/test and inflates macro-F1 by **7.7 points** (0.6605 naive vs. 0.5832 grouped 5-fold CV). Every model is trained/evaluated on the deduplicated set with an explicit no-leakage assertion
- **Neutral is the hard class:** recall 61.2%, the lowest of the three, and involved in >87% of all test-set errors — it's defined by the *absence* of polarised vocabulary rather than by vocabulary of its own
- **Topic labels are spurious, not semantic:** a fixed, ordered substring-match rule (`app`, `down`, `update`, ... → Technical_Issues; etc.) reproduces all 9,000 topic labels exactly, with 44-88% of triggers per class firing *inside* unrelated words (`happy` contains `app`). These labels should not be used for topic-based routing as-is
- **The ensemble is calibrated:** confidence buckets track observed accuracy closely across the board, so the model's own confidence score is trustworthy for routing low-confidence posts to human review
- **The learning curve hasn't plateaued:** both fast candidate models were still improving at 100% of the training data — more labelled sentiment data would likely help further

### Round 3: Signal Tracking
- **The launch soured from indifference, not approval:** net sentiment ran +0.05 before release, **-0.12 in the first 24 hours** and **-0.37 after 72 hours**. The pre-launch 95% interval is **[-0.05, +0.15]**, so the baseline is not measurably positive; the *fall* is what the data supports, and it reaches **0.43 below baseline (95% CI [0.31, 0.54])** by day three; three shifts are significant after Holm correction, including a sharp 12-hour turn at 15 Sep 00:00 (+0.16 to -0.26, p < 0.0001)
- **Not a pile-on:** 562 negative posts come from **521 distinct authors**; the ten busiest accounts are 5.6% of the corpus
- **The reused model beats a lexicon baseline:** VADER scores 0.496 macro-F1 against our **0.699**, both measured on a 140-post two-annotator consensus (Cohen's kappa **0.896**)
- **A simple monitoring rule fires 19 hours after release** with 0 pre-launch false alarms
- **The decline is not a sampling artefact:** on Mastodon alone, positives fall 57% -> 24% (p = 0.0002) and negatives rise 14% -> 41% (p = 3.6e-8)
- **Reliability, not design, drives complaints:** bugs/crashes -0.67, notifications -0.61, privacy -0.42, Siri -0.27, while UI/design is the *least* negative theme at -0.10
- **Volume and engagement peak at different moments:** the release hour is the volume peak (36 posts/h, z = 97), while the largest engagement peak came 4 days later from one Hacker News thread about *Android 17* (1,135 points, z = 183)
- **Reusing a model is not free:** the Round 2 ensemble scores **0.699 macro-F1 in this domain** versus 0.723 on tweets; applying its neutral bias (+0.021) and fixing silent truncation of 68% of posts (+0.013) mattered more than any modelling change
- **Negative posts earn more engagement** (median 2 vs 1, Kruskal-Wallis p = 0.046), repeating the "outrage engagement" pattern from Round 1

## Tech Stack

- **Languages:** Python 3.x, SQL
- **Data Processing:** pandas, numpy, sqlite3
- **Statistics:** scipy.stats (Kruskal-Wallis, Chi-square, Spearman, Mann-Whitney U, McNemar, bootstrap CIs)
- **NLP / ML:** scikit-learn (TF-IDF, logistic regression, NB-SVM), sentence-transformers (MiniLM embeddings), transformers + torch (MiniLM fine-tuning), joblib
- **Data collection:** urllib (no scraping framework), public Atom/JSON/RSS endpoints
- **Visualization:** matplotlib, seaborn
- **Reporting:** LaTeX (pdflatex)
- **Notebooks:** Jupyter
- **Database:** SQLite
