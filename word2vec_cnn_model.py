import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import joblib

# NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Tải mô hình Word2Vec và CNN model
word2vec_model = Word2Vec.load('word2vec_model.bin')  # Đảm bảo rằng bạn đã huấn luyện và lưu mô hình Word2Vec
cnn_model = joblib.load('CNN_model_using_Word2Vec.pkl')  # Đảm bảo rằng bạn đã huấn luyện và lưu mô hình CNN

def preprocess_text(text):
    """
    Tiền xử lý văn bản: loại bỏ dấu câu, số, chuyển về chữ thường,
    loại bỏ stop words và lemmatization.
    """
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ dấu câu
    text = re.sub(r'\d+', '', text)      # Loại bỏ số
    text = text.lower()                  # Chuyển về chữ thường
    words = word_tokenize(text)          # Tokenize
    words = [word for word in words if word not in stop_words]  # Loại bỏ stop words
    words = [lemmatizer.lemmatize(word) for word in words]  # Lematize các từ
    return words

def create_embedding_matrix(sentence, model, max_len=50):
    """
    Tạo embedding matrix từ câu sử dụng mô hình Word2Vec.
    """
    embedding_dim = 100  # Kích thước vector Word2Vec
    vectors = [model.wv[word] if word in model.wv else np.zeros(embedding_dim) for word in sentence]
    if len(vectors) < max_len:
        vectors += [np.zeros(embedding_dim)] * (max_len - len(vectors))  # Padding
    return np.array(vectors[:max_len])

def check_fake_news(text):
    max_len=50
    """
    Kiểm tra tin tức là thật hay giả bằng mô hình CNN và Word2Vec.
    """
    # Tiền xử lý văn bản
    processed_text = preprocess_text(text)

    # Tạo embedding matrix từ văn bản đã xử lý
    embedding = create_embedding_matrix(processed_text, word2vec_model, max_len)
    embedding = np.expand_dims(embedding, axis=0)  # Thêm chiều batch

    # Dự đoán tin tức
    prediction = cnn_model.predict(embedding)
    return "Fake news" if prediction[0][0] < 0.5 else "True news"
