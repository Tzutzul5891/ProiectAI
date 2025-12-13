import streamlit as st
from sentence_transformers import SentenceTransformer
import re

@st.cache_resource
def load_model(model_name_or_path: str, local_files_only: bool) -> SentenceTransformer:
    return SentenceTransformer(model_name_or_path, local_files_only=local_files_only)

def evaluate_semantic(user_text, correct_text):
    try:
        from config import CONFIG

        model_name_or_path = CONFIG.sbert_model_name_or_path
        local_only = CONFIG.local_models_only
    except Exception:
        model_name_or_path = "all-MiniLM-L6-v2"
        local_only = False

    user_text = user_text or ""
    correct_text = correct_text or ""

    matches = re.findall(r"L\d+-C\d+", correct_text)
    
    found_coordinates = False
    if matches:
        for match in matches:
            if match in user_text:
                found_coordinates = True
                break
    else:
        found_coordinates = True

    # Prefer local SBERT; if unavailable (e.g., offline, missing cache), fall back
    # to a deterministic lexical similarity.
    try:
        model = load_model(model_name_or_path, local_only)
        embeddings = model.encode([user_text, correct_text], normalize_embeddings=True)
        similarity = float(embeddings[0] @ embeddings[1])
        semantic_score = round(similarity * 100, 2)
    except Exception:
        from difflib import SequenceMatcher

        ratio = SequenceMatcher(None, user_text.lower(), correct_text.lower()).ratio()
        semantic_score = round(ratio * 100, 2)
    
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
