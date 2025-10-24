# LoveAI — AI Love Coach for Couple Conversation Analysis

## Overview

**LoveAI** is a research project at the intersection of NLP and couple psychology.  
Its goal is to analyze partner dialogues, detect relational problems and emotions, and provide meaningful, context-aware insights that could assist in “love coaching” or relationship support.

The system focuses on:
- detecting relational **problems** from conversation text,
- recognizing **emotions** expressed by utterances,
- experimenting with both classical ML (TF-IDF + tree models) and contextual embeddings (BERT, sentence embeddings).

## Key Components

### Datasets
- **DailyDialog** — multi-turn dialogues with intent & emotion labels (used for exploration).  
- **GoEmotions** — 58k Reddit comments annotated with 27 emotions; reduced to 7 classes for this project.  
- **Custom couple-problem dataset** — a small dataset built from online couple discussions (used for supervised problem detection).

### Preprocessing
- Text cleanup: remove emojis/URLs, normalize spacing, lemmatize, remove stop words.  
- Feature extraction: TF-IDF vectors, sentence embeddings (SBERT), contextual embeddings (BERT variants).

---

## Methods & Models

### Problem detection (topic / issue classification)
- Classical pipeline: **TF-IDF → Random Forest / Gradient Boosting** (strong baseline).  
- Topic modelling attempts: **K-Means, LDA** (results inconclusive on DailyDialog).  
- Better results obtained with **sentence embeddings + lightweight classifiers** (LogReg, RF, MLP).

Performance (examples from experiments):
- TF-IDF + Random Forest / Gradient Boosting → Accuracy ≈ 0.82 (on small domain dataset).
- Sentence embeddings + Logistic Regression / MLP → Accuracy up to ≈ 0.91 (on curated dataset).

### Emotion recognition (from text)
- Fine-tuned models on GoEmotions (mapped to 7 classes):
  - **DistilBERT**, **RoBERTa-base**, **TinyBERT**, **BERT Mini**.
- DistilBERT / RoBERTa show the best trade-off given Colab constraints (DistilBERT performed strongly).

Example results:
- DistilBERT-base: Accuracy ≈ 0.63, Weighted F1 ≈ 0.72 (training time varies per setup)
- RoBERTa-base: Accuracy ≈ 0.62, Weighted F1 ≈ 0.71

---

## Evaluation & Findings

- **TF-IDF clustering** (elbow + silhouette) failed to produce relationship-specific topics on DailyDialog — thus a domain dataset was created.  
- **Contextual embeddings** (BERT, SBERT) + simple classifiers outperform TF-IDF for capturing relational nuances.  
- Emotion detection is feasible with GoEmotions after mapping labels, but performance depends heavily on model size and fine-tuning budget.  
- Main challenges: dataset availability (privacy for couple data), conversational nuance, slang/short forms.


