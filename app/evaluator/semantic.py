import streamlit as st
import re

@st.cache_resource
def load_model(model_name_or_path: str, local_files_only: bool):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name_or_path, local_files_only=local_files_only)

def evaluate_semantic(user_text, correct_text, *, use_sbert: bool | None = None, local_models_only: bool | None = None):
    try:
        from config import CONFIG

        model_name_or_path = CONFIG.sbert_model_name_or_path
        local_only = CONFIG.local_models_only
        config_use_sbert = bool(getattr(CONFIG, "enable_sbert", True))
    except Exception:
        model_name_or_path = "all-MiniLM-L6-v2"
        local_only = False
        config_use_sbert = True

    if use_sbert is None:
        use_sbert = config_use_sbert
    if local_models_only is not None:
        local_only = bool(local_models_only)

    user_text = user_text or ""
    correct_text = correct_text or ""

    try:
        from app.utils.helpers import extract_nash_coordinates

        expected_coords = set(extract_nash_coordinates(correct_text))
        got_coords = set(extract_nash_coordinates(user_text))
        found_coordinates = (not expected_coords) or bool(expected_coords & got_coords)
    except Exception:
        matches = re.findall(r"L\\d+-C\\d+", correct_text)
        found_coordinates = (not matches) or any(match in user_text for match in matches)

    # Prefer local SBERT; if unavailable (e.g., offline, missing cache), fall back
    # to a deterministic lexical similarity.
    try:
        if not use_sbert:
            raise RuntimeError("SBERT disabled")

        model = load_model(model_name_or_path, local_only)
        embeddings = model.encode([user_text, correct_text], normalize_embeddings=True)
        similarity = float(embeddings[0] @ embeddings[1])
        semantic_score = round(similarity * 100, 2)
    except Exception:
        from difflib import SequenceMatcher

        ratio = SequenceMatcher(None, user_text.lower(), correct_text.lower()).ratio()
        semantic_score = round(ratio * 100, 2)
    
    if not found_coordinates:
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
