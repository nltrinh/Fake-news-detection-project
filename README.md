# Fake News Detection Project

## Overview

Dự án phát hiện tin giả (Fake News Detection) sử dụng các mô hình học máy và học sâu:

- **Word2Vec + Naive Bayes/CNN/RNN**
- **BERT + Naive Bayes/CNN/RNN**

Bao gồm:

- Preprocessing dữ liệu từ dataset tin thật/giả.
- Training models (code trong notebooks và .py scripts).
- Web app Flask (`app.py`) để predict tin tức.
- Visualizations (charts).

**Dataset:** News \_dataset/ (Fake.csv, True.csv) ~50-60MB each - large files warned by GitHub.

**Trained models** (excluded from repo do size):

- Word2Vec models (_.bin, _.pkl)
- BERT embeddings (\*.pkl)
- Trained classifiers (_.pkl, _.h5)

## Yêu cầu / Requirements

```bash
pip install -r requirements.txt
```

Chính: pandas, numpy, scikit-learn, tensorflow/keras, transformers (BERT), flask, gensim (Word2Vec), matplotlib/seaborn.

**Lưu ý:** Tạo virtual env:

```bash
python -m venv venv
venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
```

## Chạy dự án / Run

### 1. Web App (Flask)

```bash
python app.py
```

Mở http://127.0.0.1:5000/

- Nhập text tin tức → Predict thật/giả.

### 2. Train/Reload models (nếu cần models)

Chạy notebooks:

- Word2Vec-\*.ipynb
- BERT-\*.ipynb
  Models lưu local, không trên GitHub.

### 3. Scripts

```bash
# Example train NB with Word2Vec
python word2vec_nb_model.py
```

## Cấu trúc

```
.
├── app.py                 # Flask web app
├── requirements.txt       # Dependencies
├── .gitignore            # Excludes large files
├── News _dataset/        # CSV data
├── templates/index.html  # Web UI
├── word2vec_*.py         # Word2Vec models
├── bert_*.py             # BERT models
├── *.ipynb               # Notebooks
└── bert_chart.png, word2vec_chart.png
```

## Lưu ý / Notes

- Dataset lớn (>50MB/file) → GitHub cảnh báo, nhưng đã push.
- Để train lại models: Chạy notebooks/scripts.
- Predict chỉ cần code + dataset (models regenerate or train once).

## Contributing

Fork → Commit → PR.

© nltrinh
