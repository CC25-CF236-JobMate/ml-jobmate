inference_script = """
import pickle
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load vectorizer dan job vectors
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

job_vectors = sparse.load_npz('job_vectors.npz')
job_df = pd.read_csv('job_metadata.csv')

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\\\\s]', '', text)
    text = re.sub(r'\\\\s+', ' ', text).strip()
    return text

# Fungsi rekomendasi
def recommend_jobs(user_text, top_n=5):
    cleaned = preprocess(user_text)
    user_vec = vectorizer.transform([cleaned])
    similarities = cosine_similarity(user_vec, job_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    recommendations = job_df.iloc[top_indices].copy()
    recommendations['Similarity Score'] = similarities[top_indices]
    return recommendations[['Job Title', 'Similarity Score', 'Job Description']]

# CLI
if __name__ == '__main__':
    user_input = input("Masukkan teks resume: ")
    results = recommend_jobs(user_input)
    for idx, row in results.iterrows():
        print(f"--- Rekomendasi ---")
        print(f"Job Title        : {row['Job Title']}")
        print(f"Similarity Score : {row['Similarity Score']:.4f}")
        print(f"Job Description  : {row['Job Description'][:200]}...\\n")
"""

with open("inference_pipeline.py", "w") as f:
    f.write(inference_script)

print("✅ Created inference_pipeline.py")
