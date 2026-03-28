import joblib
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.decomposition import PCA

# Tải mô hình Naive Bayes đã huấn luyện và PCA
naive_bayes_model = joblib.load('naive_bayes_model_using_BERT.pkl')
with open('pca_model.pkl', 'rb') as f:  # Nếu bạn đã lưu PCA, nếu chưa thì cần lưu mô hình PCA
    pca = joblib.load(f)

# Tải model và tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Hàm xử lý văn bản: token hóa, tạo embedding
def embed_text(text):
    tokens = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')  # Chạy trên CPU
    with torch.no_grad():
        outputs = bert_model(**tokens)
    # Lấy trung bình của hidden states và trả về vector embedding
    return outputs.last_hidden_state.mean(dim=1).numpy()[0]

# Hàm dự đoán
def check_fake_news(text):
    # Bước 1: Làm sạch văn bản (có thể bạn đã làm sẵn trong web)
    cleaned_text = text.strip().lower()

    # Bước 2: Chuyển đổi văn bản thành embedding BERT
    embedding = embed_text(cleaned_text)
    
    # Bước 3: Giảm chiều với PCA
    embedding_reduced = pca.transform([embedding])  # PCA yêu cầu đầu vào là mảng 2D

    # Bước 4: Dự đoán với mô hình Naive Bayes
    prediction = naive_bayes_model.predict(embedding_reduced)
    
    # Bước 5: Trả về kết quả dự đoán
    return "True news" if prediction[0] == 1 else "Fake news"
