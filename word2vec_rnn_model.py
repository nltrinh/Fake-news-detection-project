import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
# Tải mô hình Word2Vec
word2vec_model = Word2Vec.load("word2vec_model.bin")

# Xử lý kích thước từ vựng
try:
    vocab_size = len(word2vec_model.wv.key_to_index)  # Với gensim 4.0+
except AttributeError:
    vocab_size = len(word2vec_model.wv.vocab)  # Với gensim cũ hơn 4.0

# Tạo từ điển word_index
word_index = {word: idx + 1 for idx, word in enumerate(word2vec_model.wv.index_to_key)}

# Tạo tokenizer
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.word_index = word_index

# Tải mô hình RNN
model = load_model("RNN_model_using_Word2Vec.h5", compile=False)

# Hàm tiền xử lý và tokenize
def preprocess_and_tokenize(text):
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ dấu câu
    text = re.sub(r'\d+', '', text)      # Loại bỏ số
    text = text.lower()                  # Chuyển về chữ thường
    words = text.split()                 # Tách từ
    sequence = tokenizer.texts_to_sequences([words])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')  # Độ dài cố định
    return padded_sequence

# Hàm kiểm tra tin giả
def check_fake_news(text):
    processed_text = preprocess_and_tokenize(text)
    prediction = model.predict(processed_text)[0][0]
    return "True news" if prediction >= 0.5 else "Fake news"
