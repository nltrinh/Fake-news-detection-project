import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

# Tải model Word2Vec và Naive Bayes
word2vec_model = Word2Vec.load('word2vec_model.bin')  # Lưu riêng file mô hình Word2Vec
naive_bayes_model = joblib.load('naive_bayes_model_using_Word2Vec.pkl')

# Tiền xử lý văn bản
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

def get_average_word2vec(words, word2vec_model):
    vec = []
    for word in words:
        if word in word2vec_model.wv:
            vec.append(word2vec_model.wv[word])
    return np.mean(vec, axis=0) if vec else np.zeros(word2vec_model.vector_size)

def check_fake_news(text):
    processed_text = preprocess_text(text)
    text_vector = get_average_word2vec(processed_text, word2vec_model).reshape(1, -1)
    prediction = naive_bayes_model.predict(text_vector)
    return "Fake news" if prediction[0] == 0 else "True news"
