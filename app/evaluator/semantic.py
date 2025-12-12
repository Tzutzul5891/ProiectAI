import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_semantic(user_text, correct_text):
    """
    Evaluează răspunsul utilizatorului comparându-l cu cel corect.
    Include verificări stricte pentru negații ("nu avem", "nu are", "nu există").
    """
    
    # 1. ANALIZA SOLUȚIEI CORECTE
    nash_matches = re.findall(r"L\d+-C\d+", correct_text)
    nash_solution_exists = len(nash_matches) > 0
    correct_is_negative = "nu există" in correct_text.lower() or "nu exista" in correct_text.lower()

    # 2. ANALIZA TEXTULUI UTILIZATORULUI
    user_text_lower = user_text.lower()
    
    # LISTA EXTINSĂ DE NEGAȚII
    negation_keywords = [
        "nu există", "nu exista", 
        "fără echilibru", "niciun echilibru", "nici un echilibru",
        "nu are echilibru", "nu are niciun", "nu are solutie",
        "nu e echilibru", "nu este echilibru",
        "nu avem echilibru", "nu avem niciun", "nu avem solutie"
    ]
    
    # Verificăm dacă userul a folosit vreuna din expresiile de mai sus
    user_claims_negative = any(kw in user_text_lower for kw in negation_keywords)

    # 3. VERIFICĂRI LOGICE (PENALIZĂRI)
    logic_penalty = False
    logic_feedback = ""

    # SCENARIUL A: Soluția există, dar userul zice că NU (folosind "nu avem", "nu are" etc.)
    if nash_solution_exists and user_claims_negative:
        logic_penalty = True
        logic_feedback = "Ai afirmat că NU avem/există echilibru, deși acesta există în realitate."

    # SCENARIUL B: Soluția există, userul zice că există, dar greșește coordonatele
    elif nash_solution_exists and not user_claims_negative:
        found_any_match = False
        for match in nash_matches:
            if match in user_text:
                found_any_match = True
                break
        
        if not found_any_match:
            logic_penalty = True
            logic_feedback = "Nu ai specificat coordonatele corecte (ex: L1-C1) în răspuns."

    # SCENARIUL C: Soluția NU există, dar userul inventează coordonate
    elif correct_is_negative and not user_claims_negative:
        if re.search(r"L\d+-C\d+", user_text):
            logic_penalty = True
            logic_feedback = "Ai identificat un echilibru (coordonate L-C), deși răspunsul corect este că nu există."

    # 4. CALCUL SCOR SEMANTIC
    model = load_model()
    embeddings = model.encode([user_text, correct_text])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    semantic_score = round(similarity * 100, 2)
    
    # 5. DECIZIA FINALĂ
    if logic_penalty:
        return 0.0, f"Răspuns incorect! {logic_feedback} (Scor anulat)."
    else:
        # Dacă logica e validă, folosim scorul semantic
        if semantic_score > 85:
            msg = "Excelent! Răspuns corect și explicație clară."
        elif semantic_score > 60:
            msg = "Răspuns bun, dar explicația poate fi formulată mai precis."
        elif semantic_score > 40:
            msg = "Răspuns acceptabil, dar destul de vag."
        else:
            msg = "Răspunsul nu pare să aibă legătură cu soluția corectă."

        return semantic_score, msg