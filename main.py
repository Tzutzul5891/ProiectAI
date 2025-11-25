import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.getcwd())

try:
    from app.modules.games import NashGame
    from app.modules.search import NQueensProblem
    from app.evaluator.semantic import evaluate_semantic
    from app.utils.pdf_generator import create_pdf
except ImportError as e:
    st.error(f"Eroare la importuri: {e}. Verifică dacă ai creat toate fișierele!")
    st.stop()

st.set_page_config(
    page_title="SmarTest - Proiect IA",
    page_icon="📝",
    layout="wide"
)

st.title("📝 SmarTest - Generator Examen & PDF")
st.markdown("Generare probleme și evaluare automată offline (fără API-uri externe).")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configurare")
    problem_type = st.radio(
        "Alege Tipul Problemei:",
        ("Jocuri (Echilibru Nash)", "Căutare (N-Queens)")
    )
    
    st.info(
        """
        **Info Proiect:**
        Aplicație locală.
        - **Backend:** Algoritmi deterministi.
        - **Evaluare:** Model SBERT + Regex.
        """
    )

if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None

if st.session_state.problem_type != problem_type:
    st.session_state.problem_type = problem_type
    st.session_state.matrix = None
    st.session_state.correct_expl = ""
    st.session_state.user_feedback = ""
    
    if problem_type == "Jocuri (Echilibru Nash)":
        st.session_state.game = NashGame()
    else:
        st.session_state.game = NQueensProblem()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("1. Generare & Export")
    
    if st.button("🎲 Generează Întrebare Nouă", use_container_width=True):
        with st.spinner("Se rulează algoritmul generator..."):
            data, explanation = st.session_state.game.generate_problem()
            st.session_state.matrix = data
            st.session_state.correct_expl = explanation
            st.session_state.user_feedback = ""
        st.success("Problemă generată cu succes!")

    if st.session_state.matrix:
        st.write("---")
        st.write("📄 **Opțiuni Export:**")
        
        if problem_type == "Jocuri (Echilibru Nash)":
            pdf_req = "Se da matricea de plati de mai jos. Identificati daca exista un Echilibru Nash pur si specificati coordonatele (ex: L1-C1)."
        else:
            pdf_req = "Pe tabla de 4x4 de mai jos, propuneti o configurare pentru 4 Regine astfel incat sa nu se atace reciproc."

        try:
            pdf_bytes = create_pdf(problem_type, pdf_req, st.session_state.matrix)
            
            st.download_button(
                label="⬇️ Descarcă Subiectul (PDF)",
                data=pdf_bytes,
                file_name="subiect_examen_ia.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"Nu s-a putut genera PDF-ul: {e}")

with col_right:
    st.subheader("2. Vizualizare și Răspuns")
    
    if st.session_state.matrix:
        if problem_type == "Jocuri (Echilibru Nash)":
            st.markdown("### Cerință:")
            st.write("Se dă matricea de plăți de mai jos. **Identifică dacă există un Echilibru Nash pur** și specifică coordonatele.")
            
            df_display = pd.DataFrame(
                st.session_state.matrix,
                index=["Linia 1", "Linia 2"],
                columns=["Coloana 1", "Coloana 2"]
            )
            st.table(df_display)
            
        else:
            st.markdown("### Cerință:")
            st.write(f"Pe tabla de **4x4** de mai jos, propune o configurare pentru **4 regine**.")
            
            df_display = pd.DataFrame(
                st.session_state.matrix,
                index=[1, 2, 3, 4],
                columns=[1, 2, 3, 4]
            )
            st.table(df_display)

        st.markdown("---")
        user_answer = st.text_area("✍️ Răspunsul tău:", height=100, placeholder="Scrie explicația aici...")

        if st.button("✅ Verifică Răspunsul", type="primary"):
            if not user_answer:
                st.warning("Te rog scrie un răspuns înainte de verificare.")
            else:
                with st.spinner("AI-ul analizează răspunsul tău..."):
                    score, feedback = evaluate_semantic(user_answer, st.session_state.correct_expl)
                
                st.markdown(f"### Scor Semantic: **{score:.2f}%**")
                
                if score > 75:
                    st.success(f"Feedback: {feedback}")
                elif score > 40:
                    st.warning(f"Feedback: {feedback}")
                else:
                    st.error(f"Feedback: {feedback}")
                
                with st.expander("🔍 Vezi Soluția Algoritmică (Gold Standard)"):
                    st.info(st.session_state.correct_expl)
    else:
        st.info("👈 Apasă pe butonul 'Generează Întrebare Nouă' din stânga pentru a începe.")