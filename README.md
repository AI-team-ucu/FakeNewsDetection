# Fake News Detection
Detect **Fake / True American news** using lightweight pipeline for fast text-classification

Primary baseline: **TF-IDF (word 1–2) + LinearSVC**


## Installation
```bash
python -m venv .env
source .env/bin/activate
pip install -U pip
pip install pandas numpy scikit-learn matplotlib wordcloud transformers nltk
```

## Data used

Cleaned dataset from Kaggle with basic Fake News Dataset:

### <a href="https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets">Dataset</a>

Preprocessed with:
* Lowercased
* Without stopwords
* No punctuation
* Lemmatized

Also was splitted into:
* **80%** - Train data
* **20%** - Validation data

## Model Pipeline

For *Vectorization* we have used **TF-IDF** with **n-grams=(1-2)**

Because it captures key phrases and down-weights boilerplate; extremely effective with linear models on text.

For *Classifier* we have used **LinearSVC** with fixed **C=1.0** and **balanced** class-weights

Because it strong accuracy/F1 on sparse TF-IDF, fast training/inference, minimal tuning.

## Evaluation

We calculated those metrics to check wether our model is good or not:

- Accuracy
- Precision Score
- Recall
- Macro F1
- Micro F1
