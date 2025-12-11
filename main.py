import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.getcwd())

try:
    from app.modules.games import NashGame
    from app.modules.search import NQueensProblem, KnightsTourProblem, TowerOfHanoiProblem
    from app.evaluator.semantic import evaluate_semantic
    from app.utils.pdf_generator import create_pdf
    from app.gui.components import (render_interactive_queens_board, board_to_text, check_queens_validity,
                                     render_interactive_knights_board, knights_board_to_text, check_knights_tour_validity,
                                     render_interactive_hanoi, hanoi_moves_to_text, check_hanoi_validity)
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
        ("Jocuri (Echilibru Nash)", "Căutare (N-Queens)", "Căutare (Turul Calului)", "Căutare (Turnurile Hanoi)")
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
    elif problem_type == "Căutare (N-Queens)":
        st.session_state.game = NQueensProblem()
    elif problem_type == "Căutare (Turul Calului)":
        st.session_state.game = KnightsTourProblem()
    else:
        st.session_state.game = TowerOfHanoiProblem()

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
        elif problem_type == "Căutare (N-Queens)":
            board_size = len(st.session_state.matrix)
            pdf_req = f"Pe tabla de {board_size}x{board_size} de mai jos, propuneti o configurare pentru regine astfel incat sa nu se atace reciproc (pe linii, coloane sau diagonale)."
        elif problem_type == "Căutare (Turul Calului)":
            board_size = len(st.session_state.matrix)
            pdf_req = f"Pe tabla de {board_size}x{board_size} de mai jos, creati un tur al calului care viziteaza fiecare casuta exact o singura data. Calul se misca in forma de 'L'."
        else:  # Tower of Hanoi
            num_disks = st.session_state.game.num_disks
            num_pegs = st.session_state.game.num_pegs
            peg_names = ["A", "B", "C", "D"][:num_pegs]
            pdf_req = f"Turnurile din Hanoi: Mutati toate cele {num_disks} discuri de pe tija {peg_names[0]} pe tija {peg_names[-1]}, respectand regulile (un disc mai mare nu poate fi plasat peste unul mai mic)."

        try:
            # For Tower of Hanoi, pass the initial state
            if problem_type == "Căutare (Turnurile Hanoi)":
                hanoi_state = st.session_state.game.initial_state
                pdf_bytes = create_pdf(problem_type, pdf_req, st.session_state.matrix, hanoi_state=hanoi_state)
            else:
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
            
        elif problem_type == "Căutare (N-Queens)":
            # Get the board size from the generated problem
            board_size = len(st.session_state.matrix)
            
            st.markdown("### Cerință:")
            st.write(f"Pe tabla de **{board_size}x{board_size}** de mai jos, plasează reginele astfel încât să nu se atace reciproc.")
            st.info("💡 **Hint:** În problema N-Regine, reginele nu trebuie să se atace pe linii, coloane sau diagonale.")
            
            # Render interactive board for N-Queens
            user_board = render_interactive_queens_board(n=board_size, key_prefix="nqueens_user")
            
            st.markdown("---")
            
            # Convert board state to text for evaluation
            user_answer = board_to_text(user_board)
            
            # Show the text representation
            with st.expander("📝 Vezi reprezentarea text a plasării tale"):
                st.write(user_answer)
        
        elif problem_type == "Căutare (Turul Calului)":
            # Get the board size and starting position
            board_size = len(st.session_state.matrix)
            start_pos = st.session_state.game.start_pos
            
            st.markdown("### Cerință:")
            st.write(f"Pe tabla de **{board_size}x{board_size}** de mai jos, creează un tur al calului care vizitează fiecare căsuță exact o dată.")
            st.info("💡 **Hint:** Calul se mișcă în formă de 'L' (2 căsuțe într-o direcție + 1 căsuță perpendicular).")
            
            # Render interactive board for Knight's Tour
            user_board = render_interactive_knights_board(n=board_size, start_pos=start_pos, key_prefix="knights_user")
            
            st.markdown("---")
            
            # Convert board state to text for evaluation
            user_answer = knights_board_to_text(user_board, start_pos)
            
            # Show the text representation
            with st.expander("📝 Vezi reprezentarea text a traseului tău"):
                st.write(user_answer)
        
        else:  # Tower of Hanoi
            num_disks = st.session_state.game.num_disks
            num_pegs = st.session_state.game.num_pegs
            initial_state = st.session_state.game.initial_state
            peg_names = ["A", "B", "C", "D"][:num_pegs]
            
            st.markdown("### Cerință:")
            st.write(f"Mută toate cele **{num_disks} discuri** de pe tija **{peg_names[0]}** pe tija **{peg_names[-1]}** folosind {num_pegs} tije.")
            st.info("💡 **Reguli:** (1) Doar un disc poate fi mutat odată. (2) Un disc mare nu poate fi plasat peste un disc mic.")
            
            # Render interactive Hanoi
            user_moves, pegs_state = render_interactive_hanoi(num_disks, num_pegs, initial_state, key_prefix="hanoi_user")
            
            st.markdown("---")
            
            # Convert moves to text
            user_answer = hanoi_moves_to_text(user_moves, num_pegs)
            
            # Show the text representation
            with st.expander("📝 Vezi lista mișcărilor tale"):
                st.write(user_answer)

        st.markdown("---")
        
        if problem_type == "Jocuri (Echilibru Nash)":
            # Keep text area for Nash equilibrium
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
        elif problem_type == "Căutare (N-Queens)":
            # N-Queens verification
            if st.button("✅ Verifică Răspunsul", type="primary"):
                # Get expected number of queens
                expected_queens = st.session_state.game.expected_queens
                
                # Check validity with detailed feedback
                is_valid, validity_msg, detailed_feedback = check_queens_validity(user_board, expected_queens)
                
                if not is_valid:
                    st.error(f"❌ {validity_msg}")
                    
                    # Show detailed feedback
                    st.markdown("### 📝 Detalii despre erori:")
                    for feedback_item in detailed_feedback:
                        st.warning(feedback_item)
                    
                    # Calculate partial score based on correctness
                    queen_count = sum(sum(row) for row in user_board)
                    board_size = len(user_board)
                    
                    # Start with base score
                    partial_score = 0
                    
                    # Give points for correct number of queens (30%)
                    if queen_count == expected_queens:
                        partial_score += 30
                    elif abs(queen_count - expected_queens) <= 2:
                        partial_score += 15
                    
                    # Semantic similarity for attempt (up to 20%)
                    with st.spinner("AI-ul analizează răspunsul tău..."):
                        semantic_score, _ = evaluate_semantic(user_answer, st.session_state.correct_expl)
                    partial_score += min(semantic_score * 0.2, 20)
                    
                    st.markdown(f"### Scor Parțial: **{partial_score:.2f}%**")
                    st.info("💡 Ai primit un scor parțial pentru încercare. Corectează erorile de mai sus și încearcă din nou!")
                else:
                    st.success(validity_msg)
                    for feedback_item in detailed_feedback:
                        st.success(feedback_item)
                    
                    # Valid configuration gets 100%
                    st.markdown(f"### Scor Final: **100.00%**")
                    st.success("Feedback: Excelent! Configurarea este perfect validă și corectă!")
                    
                with st.expander("🔍 Vezi Soluția Algoritmică (Gold Standard)"):
                    st.info(st.session_state.correct_expl)
        
        elif problem_type == "Căutare (Turul Calului)":
            # Knight's Tour verification
            if st.button("✅ Verifică Răspunsul", type="primary"):
                # Get starting position and solution
                start_pos = st.session_state.game.start_pos
                solution_board = st.session_state.game.solution_path
                
                # Check validity with detailed feedback
                is_valid, validity_msg, detailed_feedback, invalid_moves = check_knights_tour_validity(user_board, start_pos)
                
                # Use AI-based scoring
                from app.gui.components import compute_knights_tour_score
                ai_score, metrics = compute_knights_tour_score(user_board, solution_board, start_pos)
                
                if not is_valid:
                    st.error(f"❌ {validity_msg}")
                    
                    # Show detailed feedback
                    st.markdown("### 📝 Detalii despre erori:")
                    for feedback_item in detailed_feedback:
                        st.warning(feedback_item)
                    
                    # Show AI-based score with breakdown
                    st.markdown(f"### Scor AI: **{ai_score:.2f}%**")
                    
                    # Show metrics breakdown
                    with st.expander("📊 Vezi analiza detaliată AI"):
                        st.write("**Metrici de evaluare:**")
                        st.write(f"- 📍 Acoperire tabla: {metrics['coverage']*100:.1f}% (greutate 25%)")
                        st.write(f"- ✅ Mișcări valide: {metrics['valid_moves']}/{metrics['total_moves']} → {metrics['validity']*100:.1f}% (greutate 40%)")
                        st.write(f"- 🎯 Similaritate cu soluția: {metrics['similarity']*100:.1f}% (greutate 20%)")
                        st.write(f"- 🧠 Gândire strategică: {metrics['heuristic']*100:.1f}% (greutate 5%)")
                        st.write(f"- 🔗 Conectivitate traseu: {metrics['connectivity']*100:.1f}% (greutate 10%)")
                    
                    if ai_score > 50:
                        st.warning("💡 Aproape! Corectează erorile și încearcă din nou!")
                    else:
                        st.info("💡 Continuă să exersezi! Încearcă să urmezi mișcările în formă de 'L' ale calului.")
                else:
                    st.success(validity_msg)
                    for feedback_item in detailed_feedback:
                        st.success(feedback_item)
                    
                    # Valid Knight's Tour gets 100%
                    st.markdown(f"### Scor Final: **100.00%**")
                    st.success("Feedback: Excelent! Ai completat un tur valid al calului!")
                    
                    # Show perfect metrics
                    with st.expander("📊 Vezi analiza AI"):
                        st.write("**Toate metricile sunt perfecte! 🎉**")
                        st.write("- ✅ Acoperire completă")
                        st.write("- ✅ Toate mișcările sunt valide")
                        st.write("- ✅ Traseu complet și conectat")
                
                # Show solution visualization for Knight's Tour
                with st.expander("🔍 Vezi Soluția (Gold Standard)"):
                    st.info(st.session_state.correct_expl)
                    st.markdown("**Tabla cu soluția:**")
                    
                    # Create a visual representation of the solution
                    solution_display = []
                    for row in solution_board:
                        solution_display.append([str(cell) if cell >= 0 else "·" for cell in row])
                    
                    # Display as DataFrame for better formatting
                    import pandas as pd
                    df_solution = pd.DataFrame(
                        solution_display,
                        index=[f"Rând {i+1}" for i in range(len(solution_board))],
                        columns=[f"Col {i+1}" for i in range(len(solution_board))]
                    )
                    st.dataframe(df_solution, use_container_width=True)
        
        else:  # Tower of Hanoi verification
            if st.button("✅ Verifică Răspunsul", type="primary"):
                num_disks = st.session_state.game.num_disks
                num_pegs = st.session_state.game.num_pegs
                target_peg = num_pegs - 1
                solution_moves = st.session_state.game.solution_moves
                optimal_length = len(solution_moves)
                
                # Check validity
                is_complete, is_optimal, validity_msg, detailed_feedback, efficiency = check_hanoi_validity(
                    user_moves, pegs_state, num_disks, num_pegs, target_peg, optimal_length
                )
                
                if not is_complete:
                    st.warning(f"⏳ {validity_msg}")
                    
                    for feedback_item in detailed_feedback:
                        st.info(feedback_item)
                    
                    st.markdown(f"### Progres: Încă lucrezi la puzzle")
                    st.info("💡 Continuă să muți discurile! Verifică din nou când ai terminat.")
                
                elif is_complete and not is_optimal:
                    st.success(validity_msg)
                    
                    for feedback_item in detailed_feedback:
                        st.write(feedback_item)
                    
                    # Calculate score based on efficiency
                    score = 50 + (efficiency * 50)  # 50-100% based on efficiency
                    
                    st.markdown(f"### Scor: **{score:.2f}%**")
                    
                    if score >= 90:
                        st.success("Foarte bine! Aproape optim!")
                    elif score >= 70:
                        st.info("Bine! Dar poți face mai eficient.")
                    else:
                        st.warning("Completat, dar cu multe mișcări în plus. Încearcă să găsești o cale mai scurtă!")
                
                else:  # Complete and optimal
                    st.success(validity_msg)
                    for feedback_item in detailed_feedback:
                        st.success(feedback_item)
                    
                    st.markdown(f"### Scor Final: **100.00%**")
                    st.success("Feedback: Perfect! Ai rezolvat puzzle-ul cu numărul minim de mișcări!")
                
                # Show solution
                with st.expander("🔍 Vezi Soluția Optimă (Gold Standard)"):
                    st.info(st.session_state.correct_expl)
                    st.markdown(f"**Lista mișcărilor optime ({len(solution_moves)} mișcări):**")
                    
                    peg_names = ["A", "B", "C", "D"][:num_pegs]
                    move_list = [f"{i+1}. {peg_names[src]} → {peg_names[dst]}" for i, (src, dst) in enumerate(solution_moves)]
                    
                    # Show in columns for better readability
                    chunk_size = 10
                    chunks = [move_list[i:i+chunk_size] for i in range(0, len(move_list), chunk_size)]
                    
                    cols = st.columns(min(len(chunks), 3))
                    for idx, chunk in enumerate(chunks):
                        with cols[idx % 3]:
                            for move in chunk:
                                st.text(move)
    else:
        st.info("👈 Apasă pe butonul 'Generează Întrebare Nouă' din stânga pentru a începe.")