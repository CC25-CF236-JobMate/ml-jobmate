from flask import Flask, request, jsonify
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity

# Load model dan vectorizer
with open("supervised_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("job_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Flask app
app = Flask(__name__)

# Preprocessing function
def basic_preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Route untuk prediksi kategori pekerjaan
@app.route('/predict-category', methods=['POST'])
def predict_category():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'Missing input text'}), 400

    raw_text = data['text']
    processed_text = basic_preprocess(raw_text)
    vec = vectorizer.transform([processed_text])
    prediction = model.predict(vec)[0]

    return jsonify({'predicted_category': prediction})

# Untuk menjalankan lokal
if __name__ == '__main__':
    app.run(debug=True)
