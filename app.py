from flask import Flask, render_template, request, jsonify
from word2vec_nb_model import check_fake_news as word2vec_nb  
from word2vec_cnn_model import check_fake_news as word2vec_cnn
from word2vec_rnn_model import check_fake_news as word2vec_rnn
from bert_nb_model import check_fake_news as bert_nb
from bert_cnn_model import check_fake_news as bert_cnn
from bert_rnn_model import check_fake_news as bert_rnn

app = Flask(__name__)

# Trang chính
@app.route('/')
def index():
    return render_template('index.html')

# API để dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ frontend
    model_name = request.form.get('model')  # Tên model được chọn
    embedding_tool = request.form.get('embedding')  # Công cụ nhúng được chọn
    input_text = request.form.get('input_text')  # Văn bản cần dự đoán

    if not input_text.strip():
        return jsonify({"error": "Please enter some text to predict!"})

    # Gọi hàm dự đoán dựa trên model được chọn
    if model_name == 'NB'and embedding_tool == 'Word2Vec':
        prediction = word2vec_nb(input_text)
    elif model_name == 'CNN' and embedding_tool == 'Word2Vec':
        prediction = word2vec_cnn(input_text)
    elif model_name == 'RNN' and embedding_tool == 'Word2Vec':
        prediction = word2vec_rnn(input_text)
    elif model_name == 'NB' and embedding_tool == 'BERT':
        prediction = bert_nb(input_text)
    elif model_name == 'CNN' and embedding_tool == 'BERT':
        prediction = bert_cnn(input_text)
    elif model_name == 'RNN' and embedding_tool == 'BERT':
        prediction = bert_rnn(input_text)
    else:
        return jsonify({"error": "Invalid model selected!"})

    # Trả về kết quả
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
