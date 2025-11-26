import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_semantic(user_text, correct_text):
    matches = re.findall(r"L\d+-C\d+", correct_text)
    
    found_coordinates = False
    if matches:
        for match in matches:
            if match in user_text:
                found_coordinates = True
                break
    else:
        found_coordinates = True

    model = load_model()
    embeddings = model.encode([user_text, correct_text])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    semantic_score = round(similarity * 100, 2)
    
    if not found_coordinates and "L" in correct_text:
        final_score = min(semantic_score, 40)
        msg = f"Ai greșit coordonatele matematice! (Scor limitat). Deși limbajul e similar ({semantic_score}%), rezultatul e greșit."
    else:
        final_score = semantic_score
        if final_score > 80:
            msg = "Excelent! Răspuns corect și explicație clară."
        elif final_score > 60:
            msg = "Răspuns bun, dar explicația poate fi mai precisă."
        else:
            msg = "Răspunsul nu pare corect."

    return final_score, msg