import joblib
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Load mô hình CNN đã lưu
cnn_model = joblib.load('CNN_using_BERT.pkl')

# Load BERT model và tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def clean_text(text):
    """
    Làm sạch văn bản.
    """
    import re
    text = re.sub(r'\s+', ' ', text)  # Xóa khoảng trắng thừa
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Giữ lại chữ cái
    return text.strip().lower()

def preprocess_input(text):
    """
    Tiền xử lý văn bản và tạo embeddings từ BERT.
    """
    # Làm sạch văn bản
    text = clean_text(text)
    
    # Chuyển đổi văn bản thành embeddings bằng BERT
    tokens = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**tokens)
    embedding = outputs.last_hidden_state.mean(dim=1).numpy()  # Lấy trung bình embeddings
    
    # Reshape để thêm chiều phù hợp với input của CNN
    embedding = embedding[..., np.newaxis]  # Thêm chiều thứ ba (1, embedding_size, 1)
    return embedding

def check_fake_news(input_text):
    """
    Kiểm tra tin giả dựa trên đoạn văn bản đầu vào.
    """
    # Tiền xử lý văn bản đầu vào
    input_embedding = preprocess_input(input_text)
    
    # Dự đoán từ mô hình CNN
    prediction = cnn_model.predict(input_embedding)
    
    # Quy ước: tin thật (1), tin giả (0)
    return "True news" if prediction > 0.5 else "Fake news"
