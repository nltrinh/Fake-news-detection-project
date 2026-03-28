import numpy as np
import re
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, BertModel
import torch

# 1. Hàm làm sạch văn bản
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Xóa khoảng trắng thừa
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Giữ lại chữ cái
    return text.strip().lower()

# 2. Load mô hình BERT
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# 3. Hàm nhúng văn bản bằng BERT
def embed_text(text):
    tokens = bert_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# 4. Load mô hình RNN đã huấn luyện
rnn_model = load_model('RNN_model_using_BERT.h5', compile=False)

# 5. Hàm kiểm tra tin tức thật hay giả
def check_fake_news(text):
    # Làm sạch và nhúng văn bản
    clean_text_input = clean_text(text)
    embedding = embed_text([clean_text_input])[0]
    # Dự đoán
    prediction = rnn_model.predict(np.array([embedding]))
    return "True news" if prediction[0] > 0.5 else "Fake news"