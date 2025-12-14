import streamlit as st
import pandas as pd
from dataclasses import replace
import random
import sys
import os

sys.path.append(os.getcwd())

try:
    from config import CONFIG
    from app.utils.helpers import set_global_seed

    set_global_seed(CONFIG.seed)
except Exception:
    # Config is optional; app should still start without it.
    pass

try:
    from app.modules.games import NashGame
    from app.modules.search import NQueensProblem, KnightsTourProblem, TowerOfHanoiProblem
    from app.modules.graph_coloring import GraphColoringProblem, parse_coloring_text, evaluate_graph_coloring
    from app.modules.csp import CSPInstanceProblem, list_csp_instances
    from app.modules.adversarial import AlphaBetaTreeProblem, list_adversarial_trees
    from app.modules.strategy_choice import StrategyChoiceProblem
    from app.evaluator.adversarial import evaluate_alpha_beta_answer
    from app.evaluator.csp import evaluate_csp_backtracking_answer
    from app.evaluator.semantic import evaluate_semantic
    from app.evaluator.strategy_choice import evaluate_strategy_choice
    from app.models import TestSession
    from app.utils.helpers import (
        extract_csp_var_value_pairs,
        extract_graph_coloring_mapping,
        extract_minmax_value_and_leaves,
        extract_nash_coordinates,
        format_csp_pairs,
        format_graph_coloring_mapping,
    )
    from app.utils.pdf_generator import (
        create_evaluation_pdf,
        create_pdf,
        create_test_evaluation_pdf,
        create_test_pdf,
    )
    from app.utils.pdf_parser import extract_text_from_pdf
    from app.gui.components import (render_interactive_queens_board, board_to_text, check_queens_validity,
                                     render_interactive_knights_board, knights_board_to_text, check_knights_tour_validity,
                                     render_interactive_hanoi, hanoi_moves_to_text, check_hanoi_validity,
                                     render_interactive_graph_coloring, graph_coloring_to_text)
except ImportError as e:
    st.error(f"Eroare la importuri: {e}. Verifică dacă ai creat toate fișierele!")
    st.stop()

UI_LABEL_NASH = "Jocuri (Echilibru Nash)"
UI_LABEL_NQUEENS = "Căutare (N-Queens)"
UI_LABEL_KNIGHTS = "Căutare (Turul Calului)"
UI_LABEL_HANOI = "Căutare (Turnurile Hanoi)"
UI_LABEL_GRAPH_COLORING = "CSP (Graph Coloring)"
UI_LABEL_CSP_BT = "CSP (BT + FC/MRV/AC-3)"
UI_LABEL_STRATEGY = "Teorie (Alegere Strategie)"
UI_LABEL_ADVERSARIAL = "Adversarial (MinMax + Alpha-Beta)"


def build_prompt_text(ui_label: str, *, data, metadata: dict, fallback_game=None) -> str:
    """Keep prompts consistent with existing single-question UX/PDF wording."""

    if ui_label == UI_LABEL_NASH:
        return (
            "Se da matricea de plati de mai jos. Identificati daca exista un Echilibru Nash pur si "
            "specificati coordonatele (ex: L1-C1)."
        )

    if ui_label == UI_LABEL_NQUEENS:
        board_size = len(data) if data else 0
        return (
            f"Pe tabla de {board_size}x{board_size} de mai jos, propuneti o configurare pentru regine "
            "astfel incat sa nu se atace reciproc (pe linii, coloane sau diagonale)."
        )

    if ui_label == UI_LABEL_KNIGHTS:
        board_size = len(data) if data else 0
        return (
            f"Pe tabla de {board_size}x{board_size} de mai jos, creati un tur al calului care viziteaza "
            "fiecare casuta exact o singura data. Calul se misca in forma de 'L'."
        )

    if ui_label == UI_LABEL_HANOI:
        num_disks = metadata.get("num_disks") or getattr(fallback_game, "num_disks", None) or "?"
        num_pegs = metadata.get("num_pegs") or getattr(fallback_game, "num_pegs", None) or 3
        peg_names = ["A", "B", "C", "D"][: int(num_pegs)]
        return (
            f"Turnurile din Hanoi: Mutati toate cele {num_disks} discuri de pe tija {peg_names[0]} pe "
            f"tija {peg_names[-1]}, respectand regulile (un disc mai mare nu poate fi plasat peste unul mai mic)."
        )

    if ui_label == UI_LABEL_GRAPH_COLORING:
        n = metadata.get("n") or (len(data) if data else "?")
        k = metadata.get("k") or len(metadata.get("color_names") or []) or "?"
        colors = metadata.get("color_names") or []
        colors_text = ", ".join(map(str, colors)) if colors else "—"
        return (
            "Se dă un graf neorientat (noduri 1..n) reprezentat prin matricea de adiacență de mai jos. "
            f"Colorați nodurile folosind cel mult k={k} culori astfel încât două noduri adiacente să nu "
            f"aibă aceeași culoare. Culori permise: {colors_text}. (n={n})"
        )

    if ui_label == UI_LABEL_ADVERSARIAL:
        return (
            "Cerința 4 — MinMax + Alpha-Beta: Pentru arborele dat, calculează valoarea din rădăcină și "
            "numărul de frunze evaluate (vizitate efectiv) de algoritmul Alpha-Beta, în parcurgere stânga→dreapta."
        )

    return ""


TEST_TOPIC_REGISTRY = {
    "Nash (Echilibru Nash 2x2)": {
        "chapter": "Games",
        "ui_label": UI_LABEL_NASH,
        "factory": NashGame,
    },
    "N-Queens": {
        "chapter": "Search",
        "ui_label": UI_LABEL_NQUEENS,
        "factory": NQueensProblem,
    },
    "Turul Calului": {
        "chapter": "Search",
        "ui_label": UI_LABEL_KNIGHTS,
        "factory": KnightsTourProblem,
    },
    "Turnurile Hanoi": {
        "chapter": "Search",
        "ui_label": UI_LABEL_HANOI,
        "factory": TowerOfHanoiProblem,
    },
    "Graph Coloring": {
        "chapter": "CSP",
        "ui_label": UI_LABEL_GRAPH_COLORING,
        "factory": GraphColoringProblem,
    },
    "Alegere Strategie (Cerința 1)": {
        "chapter": "Search",
        "ui_label": UI_LABEL_STRATEGY,
        "factory": StrategyChoiceProblem,
    },
    "MinMax + Alpha-Beta (Cerința 4)": {
        "chapter": "Adversarial",
        "ui_label": UI_LABEL_ADVERSARIAL,
        "factory": AlphaBetaTreeProblem,
    },
}

def is_test_answered(question, answer) -> bool:
    ui_label = (question.metadata or {}).get("ui_label") or question.type

    if ui_label == UI_LABEL_NASH:
        if isinstance(answer, str):
            return bool(answer.strip())
        return bool(str(answer).strip()) if answer is not None else False

    if ui_label == UI_LABEL_NQUEENS:
        if isinstance(answer, dict):
            board = answer.get("board")
        else:
            board = None
        if not board:
            return False
        return any(any(bool(cell) for cell in row) for row in board)

    if ui_label == UI_LABEL_KNIGHTS:
        if isinstance(answer, dict):
            board = answer.get("board")
        else:
            board = None
        if not board:
            return False
        # Consider answered if user placed at least one move beyond the start (0).
        return any(cell > 0 for row in board for cell in row)

    if ui_label == UI_LABEL_HANOI:
        if isinstance(answer, dict):
            moves = answer.get("moves") or []
        else:
            moves = []
        return len(moves) > 0

    if ui_label == UI_LABEL_GRAPH_COLORING:
        if isinstance(answer, dict):
            assignment = answer.get("assignment") or {}
            text = str(answer.get("text") or "")
            return bool(assignment) or bool(text.strip())
        if isinstance(answer, str):
            return bool(answer.strip())
        return False

    if ui_label == UI_LABEL_STRATEGY:
        if isinstance(answer, dict):
            chosen = str(answer.get("strategy") or answer.get("strategy_label") or "").strip()
            return bool(chosen)
        if isinstance(answer, str):
            return bool(answer.strip())
        return False

    if ui_label == UI_LABEL_ADVERSARIAL:
        if isinstance(answer, dict):
            root_val = str(answer.get("root_value") or "").strip()
            leaves = str(answer.get("visited_leaves") or "").strip()
            return bool(root_val) or bool(leaves)
        if isinstance(answer, str):
            return bool(answer.strip())
        return False

    return bool(answer)


def evaluate_test_question(question, answer) -> tuple[float, str, dict]:
    """Return (score, message, details) for a test question attempt."""

    ui_label = (question.metadata or {}).get("ui_label") or question.type

    if ui_label == UI_LABEL_NASH:
        user_text = ""
        if isinstance(answer, str):
            user_text = answer
        elif isinstance(answer, dict):
            user_text = str(answer.get("text") or "")

        if not user_text.strip():
            return 0.0, "Necompletat.", {}

        score, feedback = evaluate_semantic(user_text, question.correct_explanation)
        return float(score), str(feedback), {}

    if ui_label == UI_LABEL_NQUEENS:
        board = answer.get("board") if isinstance(answer, dict) else None
        if not board:
            return 0.0, "Necompletat.", {}

        expected_queens = (question.metadata or {}).get("expected_queens") or len(board)
        is_valid, validity_msg, detailed_feedback = check_queens_validity(board, expected_queens)

        if is_valid:
            return 100.0, str(validity_msg), {"details": detailed_feedback}

        queen_count = sum(sum(bool(cell) for cell in row) for row in board)
        partial_score = 0.0
        if queen_count == expected_queens:
            partial_score += 30
        elif abs(queen_count - expected_queens) <= 2:
            partial_score += 15

        semantic_score, _ = evaluate_semantic(board_to_text(board), question.correct_explanation)
        partial_score += min(float(semantic_score) * 0.2, 20.0)

        return float(partial_score), str(validity_msg), {"details": detailed_feedback}

    if ui_label == UI_LABEL_KNIGHTS:
        board = answer.get("board") if isinstance(answer, dict) else None
        if not board:
            return 0.0, "Necompletat.", {}

        start_pos = (question.metadata or {}).get("start_pos") or (0, 0)
        solution_board = question.correct_answer

        is_valid, validity_msg, detailed_feedback, _invalid_moves = check_knights_tour_validity(board, start_pos)
        from app.gui.components import compute_knights_tour_score

        ai_score, metrics = compute_knights_tour_score(board, solution_board, start_pos)
        score = 100.0 if is_valid else float(ai_score)
        return score, str(validity_msg), {"details": detailed_feedback, "metrics": metrics}

    if ui_label == UI_LABEL_HANOI:
        if not isinstance(answer, dict):
            return 0.0, "Necompletat.", {}

        moves = answer.get("moves") or []
        pegs_state = answer.get("pegs_state") or {}

        if len(moves) == 0:
            return 0.0, "Necompletat.", {}

        num_disks = (question.metadata or {}).get("num_disks") or 0
        num_pegs = (question.metadata or {}).get("num_pegs") or 3
        target_peg = int(num_pegs) - 1
        solution_moves = question.correct_answer or []
        optimal_length = len(solution_moves)

        is_complete, is_optimal, validity_msg, detailed_feedback, efficiency = check_hanoi_validity(
            moves, pegs_state, num_disks, num_pegs, target_peg, optimal_length
        )

        if not is_complete:
            return 0.0, str(validity_msg), {"details": detailed_feedback, "efficiency": efficiency}

        if is_optimal:
            return 100.0, str(validity_msg), {"details": detailed_feedback, "efficiency": 1.0}

        score = 50.0 + (float(efficiency) * 50.0)
        return float(score), str(validity_msg), {"details": detailed_feedback, "efficiency": efficiency}

    if ui_label == UI_LABEL_GRAPH_COLORING:
        metadata = question.metadata or {}
        n = int(metadata.get("n") or (len(question.data) if question.data else 0) or 0)
        color_names = list(metadata.get("color_names") or [])
        edges = list(metadata.get("edges") or [])

        assignment: dict[int, str] = {}
        parse_errors: list[str] = []

        if isinstance(answer, dict):
            assignment = dict(answer.get("assignment") or {})
            text = str(answer.get("text") or "")
            if not assignment and text.strip():
                parsed = parse_coloring_text(text, n=n, color_names=color_names)
                assignment = parsed.assignment
                parse_errors = parsed.errors
        elif isinstance(answer, str):
            parsed = parse_coloring_text(answer, n=n, color_names=color_names)
            assignment = parsed.assignment
            parse_errors = parsed.errors
        else:
            return 0.0, "Necompletat.", {}

        score, message, details = evaluate_graph_coloring(
            assignment,
            n=n,
            edges=edges,
            color_names=color_names,
        )
        if parse_errors:
            details = dict(details or {})
            details["parse_errors"] = parse_errors
            message = f"{message} • Probleme parsare: {len(parse_errors)}"
        return float(score), str(message), details

    if ui_label == UI_LABEL_STRATEGY:
        if not isinstance(answer, dict):
            # Best-effort: allow string answers (treat as chosen strategy).
            answer = {"strategy": str(answer or ""), "justification": ""}

        chosen = str(answer.get("strategy") or answer.get("strategy_label") or "").strip()
        justification = str(answer.get("justification") or answer.get("text") or "").strip()

        grading = (question.metadata or {}).get("strategy_choice") or {}
        score, message, details = evaluate_strategy_choice(
            chosen,
            justification,
            correct_strategy=str(question.correct_answer or ""),
            grading=grading if isinstance(grading, dict) else {},
        )
        return float(score), str(message), details

    if ui_label == UI_LABEL_ADVERSARIAL:
        root_value_text = ""
        visited_leaves_text = ""

        if isinstance(answer, dict):
            root_value_text = str(answer.get("root_value") or "")
            visited_leaves_text = str(answer.get("visited_leaves") or "")
        elif isinstance(answer, str):
            parts = [p.strip() for p in str(answer).replace(",", " ").split() if p.strip()]
            if len(parts) >= 1:
                root_value_text = parts[0]
            if len(parts) >= 2:
                visited_leaves_text = parts[1]

        expected = question.correct_answer if isinstance(question.correct_answer, dict) else None
        score, message, details = evaluate_alpha_beta_answer(
            root_value_text,
            visited_leaves_text,
            expected=expected,
        )
        return float(score), str(message), details

    return 0.0, "Tip de întrebare necunoscut.", {}


def render_debug_panel() -> None:
    question = st.session_state.get("question")
    st.markdown("---")
    with st.expander("🔧 Debug: `Question` / `TestSession`", expanded=True):
        if not question:
            st.info("Nu există încă o întrebare generată (`st.session_state.question` e None).")
        else:
            st.write("**Question (dict complet):**")
            st.json(question.to_dict(include_answer_key=True))
            st.write("**Question (fără answer key):**")
            st.json(question.to_public_dict())
            st.write("**Answer key (doar cheie):**")
            st.json(question.to_answer_key_dict())

        st.write("**TestSession.export_test():**")
        st.json(st.session_state.test_session.export_test())
        st.write("**TestSession.export_answer_key():**")
        st.json(st.session_state.test_session.export_answer_key())


def render_test_results(session: TestSession) -> None:
    st.subheader("✅ Rezultate Test")

    total = len(session.questions)
    answered = sum(1 for q in session.questions if is_test_answered(q, session.user_answers.get(q.id)))

    results_rows = []
    feedback_by_id: dict[str, str] = {}
    total_score = 0.0
    for idx, q in enumerate(session.questions, start=1):
        topic = (q.metadata or {}).get("topic") or (q.metadata or {}).get("ui_label") or q.type
        answer = session.user_answers.get(q.id)
        attempted = is_test_answered(q, answer)
        score, message, _details = evaluate_test_question(q, answer if attempted else None)
        session.scores[q.id] = float(score)
        feedback_by_id[q.id] = str(message)
        total_score += float(score)
        results_rows.append(
            {
                "#": idx,
                "Subiect": topic,
                "Completat": "da" if attempted else "nu",
                "Scor": round(float(score), 2),
                "Feedback": message,
            }
        )

    st.session_state.test_session = session
    avg_score = (total_score / total) if total > 0 else 0.0
    st.info(f"Răspunsuri completate: {answered}/{total} • Scor mediu: {avg_score:.2f}%")

    try:
        st.session_state.test_eval_pdf_bytes = create_test_evaluation_pdf(
            {
                "questions": session.questions,
                "user_answers": session.user_answers,
                "scores": session.scores,
                "feedback": feedback_by_id,
            }
        )
    except Exception as e:
        st.session_state.test_eval_pdf_bytes = None
        st.warning(f"Nu s-a putut genera PDF-ul de evaluare: {e}")

    col_back, col_pdf_test, col_pdf_key, col_pdf_eval = st.columns([1, 1, 1, 1])
    with col_back:
        if st.button("⬅️ Înapoi la întrebări", key="test_back_to_questions"):
            st.session_state.test_view = "questions"
            st.rerun()
    with col_pdf_test:
        if st.session_state.get("test_pdf_bytes"):
            st.download_button(
                "⬇️ Descarcă Test (PDF)",
                data=st.session_state["test_pdf_bytes"],
                file_name="test_ia.pdf",
                mime="application/pdf",
                key="dl_test_pdf_results",
                use_container_width=True,
            )
    with col_pdf_key:
        if st.session_state.get("answer_key_pdf_bytes"):
            st.download_button(
                "⬇️ Descarcă Answer Key (PDF)",
                data=st.session_state["answer_key_pdf_bytes"],
                file_name="answer_key_ia.pdf",
                mime="application/pdf",
                key="dl_answer_key_pdf_results",
                use_container_width=True,
            )
    with col_pdf_eval:
        if st.session_state.get("test_eval_pdf_bytes"):
            st.download_button(
                "⬇️ Descarcă Evaluare (PDF)",
                data=st.session_state["test_eval_pdf_bytes"],
                file_name="evaluare_test_ia.pdf",
                mime="application/pdf",
                key="dl_test_eval_pdf_results",
                use_container_width=True,
            )

    st.dataframe(pd.DataFrame(results_rows), use_container_width=True, hide_index=True)

    show_key = st.checkbox("Arată și answer key în aplicație", value=False, key="show_answer_key_in_app")
    if show_key:
        for idx, q in enumerate(session.questions, start=1):
            topic = (q.metadata or {}).get("topic") or (q.metadata or {}).get("ui_label") or q.type
            with st.expander(f"Întrebarea {idx}: {topic}"):
                st.write(q.prompt_text)
                st.markdown("**Răspunsul tău:**")
                st.write(session.user_answers.get(q.id, ""))
                st.markdown("**Răspuns corect:**")
                st.write(q.correct_answer)
                st.markdown("**Explicație:**")
                st.write(q.correct_explanation)

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
    app_mode = st.radio("Mod:", ("O singură întrebare", "Test (N întrebări)"), key="app_mode_select")

    generate_test_clicked = False
    selected_test_topics: list[str] = []
    test_num_questions = 0
    test_seed = 42

    if app_mode == "O singură întrebare":
        problem_type = st.radio(
            "Alege Tipul Problemei:",
            (
                UI_LABEL_NASH,
                UI_LABEL_NQUEENS,
                UI_LABEL_KNIGHTS,
                UI_LABEL_HANOI,
                UI_LABEL_GRAPH_COLORING,
                UI_LABEL_CSP_BT,
                UI_LABEL_ADVERSARIAL,
            ),
            key="single_problem_type",
        )
    else:
        st.markdown("### 🧪 Configurare Test")
        selected_chapters = st.multiselect(
            "Capitole:",
            ("Search", "Games", "CSP", "Adversarial"),
            default=("Search", "Games", "CSP"),
            key="test_chapters",
        )

        if "Adversarial" in selected_chapters:
            st.caption("Adversarial: disponibil MinMax + Alpha-Beta (Cerința 4).")
        if "CSP" in selected_chapters:
            st.caption("CSP: disponibil Graph Coloring.")

        available_topics = [
            name for name, cfg in TEST_TOPIC_REGISTRY.items() if cfg["chapter"] in set(selected_chapters)
        ]
        selected_test_topics = st.multiselect(
            "Subiecte:",
            options=available_topics,
            default=available_topics,
            key="test_topics",
        )

        test_num_questions = st.slider(
            "Număr întrebări:",
            min_value=3,
            max_value=10,
            value=3,
            step=1,
            key="test_num_questions",
        )

        try:
            default_seed = CONFIG.seed if CONFIG.seed is not None else 42
        except Exception:
            default_seed = 42
        test_seed = int(
            st.number_input(
                "Seed (random controlat):",
                min_value=0,
                max_value=1_000_000_000,
                value=int(default_seed),
                step=1,
                key="test_seed",
            )
        )

        generate_test_clicked = st.button("🧪 Generează Test", use_container_width=True)

        if st.session_state.get("test_pdf_bytes"):
            st.download_button(
                "⬇️ Descarcă Test (PDF)",
                data=st.session_state["test_pdf_bytes"],
                file_name="test_ia.pdf",
                mime="application/pdf",
                key="dl_test_pdf_sidebar",
                use_container_width=True,
            )

        if st.session_state.get("answer_key_pdf_bytes"):
            st.download_button(
                "⬇️ Descarcă Answer Key (PDF)",
                data=st.session_state["answer_key_pdf_bytes"],
                file_name="answer_key_ia.pdf",
                mime="application/pdf",
                key="dl_answer_key_pdf_sidebar",
                use_container_width=True,
            )
    
    st.info(
        """
        **Info Proiect:**
        Aplicație locală.
        - **Backend:** Algoritmi deterministi.
        - **Evaluare:** Model SBERT + Regex.
        """
    )
    debug_mode = st.checkbox("🔧 Debug (arată `Question`/`TestSession`)", value=False)

if "app_mode" not in st.session_state:
    st.session_state.app_mode = None

if st.session_state.app_mode != app_mode:
    st.session_state.app_mode = app_mode
    st.session_state.matrix = None
    st.session_state.correct_expl = ""
    st.session_state.user_feedback = ""
    st.session_state.question = None
    st.session_state.test_session = TestSession()
    st.session_state.test_pdf_bytes = None
    st.session_state.answer_key_pdf_bytes = None
    st.session_state.test_eval_pdf_bytes = None
    st.session_state.test_view = "questions"

if app_mode == "Test (N întrebări)":
    if "test_session" not in st.session_state:
        st.session_state.test_session = TestSession()
    if "test_view" not in st.session_state:
        st.session_state.test_view = "questions"

    if generate_test_clicked:
        st.session_state.test_pdf_bytes = None
        st.session_state.answer_key_pdf_bytes = None
        st.session_state.test_eval_pdf_bytes = None
        st.session_state.test_generation_errors = []
        st.session_state.test_view = "questions"

        if not selected_test_topics:
            st.error("Selectează cel puțin un subiect pentru test.")
        else:
            with st.spinner("Se generează testul..."):
                try:
                    from app.utils.helpers import set_global_seed

                    set_global_seed(test_seed)
                except Exception:
                    random.seed(test_seed)

                rng = random.Random(test_seed)
                if test_num_questions <= len(selected_test_topics):
                    planned_topics = rng.sample(selected_test_topics, k=test_num_questions)
                else:
                    planned_topics = [rng.choice(selected_test_topics) for _ in range(test_num_questions)]

                questions = []
                errors = []
                attempts_by_index: dict[int, int] = {}
                attempts = 0
                max_attempts = test_num_questions * 5

                while len(questions) < test_num_questions and attempts < max_attempts:
                    q_index = len(questions) + 1
                    attempt_no = attempts_by_index.get(q_index, 0) + 1
                    attempts_by_index[q_index] = attempt_no

                    if attempt_no == 1:
                        topic_name = planned_topics[q_index - 1]
                    else:
                        topic_name = rng.choice(selected_test_topics)

                    cfg = TEST_TOPIC_REGISTRY[topic_name]
                    gen = cfg["factory"]()
                    try:
                        q = gen.generate_question(
                            ui_label=cfg["ui_label"],
                            chapter=cfg["chapter"],
                            extra_metadata={"topic": topic_name, "seed": test_seed, "index": q_index, "attempt": attempt_no},
                        )
                        if not q.data:
                            raise ValueError("generator returned empty data")
                        prompt_text = build_prompt_text(
                            cfg["ui_label"],
                            data=q.data,
                            metadata=q.metadata,
                            fallback_game=gen,
                        )
                        if prompt_text:
                            q = replace(q, prompt_text=prompt_text)
                        questions.append(q)
                    except Exception as e:
                        errors.append(f"Q{q_index} ({topic_name}, incercarea {attempt_no}): {e}")
                    finally:
                        attempts += 1

                st.session_state.test_generation_errors = errors
                st.session_state.test_session = TestSession(questions=questions, current_index=0)
                st.session_state.question = st.session_state.test_session.current_question

                if questions:
                    st.session_state.test_pdf_bytes = create_test_pdf(questions, include_answer_key=False)
                    st.session_state.answer_key_pdf_bytes = create_test_pdf(questions, include_answer_key=True)

    st.subheader("🧪 Modul Test")

    col_pdf_test, col_pdf_key = st.columns(2)
    with col_pdf_test:
        if st.session_state.get("test_pdf_bytes"):
            st.download_button(
                "⬇️ Descarcă Test (PDF)",
                data=st.session_state["test_pdf_bytes"],
                file_name="test_ia.pdf",
                mime="application/pdf",
                key="dl_test_pdf_main",
                use_container_width=True,
            )
    with col_pdf_key:
        if st.session_state.get("answer_key_pdf_bytes"):
            st.download_button(
                "⬇️ Descarcă Answer Key (PDF)",
                data=st.session_state["answer_key_pdf_bytes"],
                file_name="answer_key_ia.pdf",
                mime="application/pdf",
                key="dl_answer_key_pdf_main",
                use_container_width=True,
            )

    session = st.session_state.test_session
    if not session.questions:
        st.info("Configurează testul în sidebar și apasă „Generează Test”.")
    else:
        if st.session_state.test_view == "results":
            render_test_results(session)
            if debug_mode:
                render_debug_panel()
            st.stop()

        current = session.current_question
        st.session_state.question = current

        st.markdown(f"### Întrebarea {session.current_index + 1}/{len(session.questions)}")
        st.write(current.prompt_text)

        ui_label = (current.metadata or {}).get("ui_label") or current.type
        if ui_label == UI_LABEL_NASH:
            df_display = pd.DataFrame(
                current.data,
                index=["Linia 1", "Linia 2"],
                columns=["Coloana 1", "Coloana 2"],
            )
            st.table(df_display)
        elif ui_label == UI_LABEL_HANOI:
            initial_state = (current.metadata or {}).get("initial_state")
            if initial_state:
                st.write("Stare inițială (tije):")
                peg_names = ["A", "B", "C", "D"][: len(initial_state)]
                cols = st.columns(len(initial_state))
                for peg_idx in range(len(initial_state)):
                    with cols[peg_idx]:
                        st.write(f"Tija {peg_names[peg_idx]}")
                        st.write(initial_state.get(peg_idx, []))
            else:
                st.write(current.data)
        elif ui_label == UI_LABEL_GRAPH_COLORING:
            metadata = current.metadata or {}
            n = int(metadata.get("n") or (len(current.data) if current.data else 0) or 0)
            k = int(metadata.get("k") or len(metadata.get("color_names") or []) or 0)
            color_names = list(metadata.get("color_names") or [])
            edges = list(metadata.get("edges") or [])

            if color_names:
                st.caption(f"n={n} • k={k} • culori: {', '.join(map(str, color_names))}")
            else:
                st.caption(f"n={n} • k={k}")

            with st.expander("Muchii (edge list)"):
                st.write(", ".join(f"({u},{v})" for u, v in edges) if edges else "—")

            try:
                df_board = pd.DataFrame(
                    current.data,
                    index=[str(i) for i in range(1, n + 1)] if n else None,
                    columns=[str(i) for i in range(1, n + 1)] if n else None,
                )
                st.dataframe(df_board, use_container_width=True)
            except Exception:
                st.write(current.data)
        elif ui_label == UI_LABEL_STRATEGY:
            try:
                df_info = pd.DataFrame(current.data, columns=["Câmp", "Valoare"])
                st.table(df_info)
            except Exception:
                st.write(current.data)
        elif ui_label == UI_LABEL_ADVERSARIAL:
            st.caption("Parcurgere: stânga → dreapta (ordinea copiilor din arbore).")
            try:
                lines: list[str] = []
                for row in (current.data or []):
                    if isinstance(row, (list, tuple)) and row:
                        lines.append(str(row[0]))
                    else:
                        lines.append(str(row))
                st.code("\n".join(lines), language="text")
            except Exception:
                st.write(current.data)
        else:
            try:
                df_board = pd.DataFrame(current.data)
                st.dataframe(df_board, use_container_width=True)
            except Exception:
                st.write(current.data)

        answer_key = f"test_answer_{current.id}"
        if answer_key not in st.session_state:
            existing_answer = session.user_answers.get(current.id, "")
            if isinstance(existing_answer, dict):
                st.session_state[answer_key] = str(existing_answer.get("text") or "")
            else:
                st.session_state[answer_key] = existing_answer

        ui_label = (current.metadata or {}).get("ui_label") or current.type

        if ui_label == UI_LABEL_NASH:
            user_answer = st.text_area("✍️ Răspunsul tău:", key=answer_key, height=140)
            session.user_answers[current.id] = user_answer

        elif ui_label == UI_LABEL_NQUEENS:
            board_size = (current.metadata or {}).get("n") or len(current.data)
            user_board = render_interactive_queens_board(n=board_size, key_prefix=f"test_{current.id}_nqueens")
            user_text = board_to_text(user_board)
            with st.expander("📝 Vezi reprezentarea text a plasării tale"):
                st.write(user_text)
            session.user_answers[current.id] = {"board": user_board, "text": user_text}

        elif ui_label == UI_LABEL_KNIGHTS:
            board_size = (current.metadata or {}).get("n") or len(current.data)
            start_pos = (current.metadata or {}).get("start_pos") or (0, 0)
            user_board = render_interactive_knights_board(
                n=board_size, start_pos=start_pos, key_prefix=f"test_{current.id}_knights"
            )
            user_text = knights_board_to_text(user_board, start_pos)
            with st.expander("📝 Vezi reprezentarea text a traseului tău"):
                st.write(user_text)
            session.user_answers[current.id] = {"board": user_board, "text": user_text}

        elif ui_label == UI_LABEL_HANOI:
            num_disks = (current.metadata or {}).get("num_disks")
            num_pegs = (current.metadata or {}).get("num_pegs")
            initial_state = (current.metadata or {}).get("initial_state")
            if num_disks is None or num_pegs is None or initial_state is None:
                st.error("Date lipsă pentru Turnurile din Hanoi.")
            else:
                user_moves, pegs_state = render_interactive_hanoi(
                    num_disks, num_pegs, initial_state, key_prefix=f"test_{current.id}_hanoi"
                )
                user_text = hanoi_moves_to_text(user_moves, num_pegs)
                with st.expander("📝 Vezi lista mișcărilor tale"):
                    st.write(user_text)
                session.user_answers[current.id] = {
                    "moves": user_moves,
                    "pegs_state": pegs_state,
                    "text": user_text,
                }
        elif ui_label == UI_LABEL_GRAPH_COLORING:
            metadata = current.metadata or {}
            n = int(metadata.get("n") or (len(current.data) if current.data else 0) or 0)
            color_names = list(metadata.get("color_names") or [])

            input_mode = st.radio(
                "Mod răspuns:",
                ("Dropdown", "Text"),
                horizontal=True,
                key=f"test_{current.id}_graph_mode",
            )

            if input_mode == "Text":
                user_text = st.text_area(
                    "✍️ Răspunsul tău (ex: 1:R, 2:G, 3:B):",
                    key=answer_key,
                    height=120,
                )
                parsed = parse_coloring_text(user_text, n=n, color_names=color_names)
                if parsed.errors:
                    with st.expander("⚠️ Probleme de parsare"):
                        for err in parsed.errors:
                            st.write(f"- {err}")
                session.user_answers[current.id] = {
                    "assignment": parsed.assignment,
                    "text": user_text,
                    "mode": "text",
                }
            else:
                assignment = render_interactive_graph_coloring(
                    n=n,
                    color_names=color_names,
                    key_prefix=f"test_{current.id}_graph",
                )
                user_text = graph_coloring_to_text(assignment)
                with st.expander("📝 Vezi răspunsul tău (text)"):
                    st.write(user_text or "Nu ai asignat încă nicio culoare.")
                session.user_answers[current.id] = {
                    "assignment": assignment,
                    "text": user_text,
                    "mode": "dropdown",
                }
        elif ui_label == UI_LABEL_STRATEGY:
            grading = (current.metadata or {}).get("strategy_choice") or {}
            allowed = list(grading.get("allowed_strategies") or []) if isinstance(grading, dict) else []

            st.info(
                "Întrebarea asta este una de TEORIE: nu rezolvi instanța efectiv, ci alegi strategia potrivită.\n"
                "Alegerea strategiei îți afectează DOAR scorul (nu schimbă generatorul/algoritmii).\n"
                "Scor: 100% pe match exact al strategiei, parțial pentru opțiuni „aproape”. "
                "Justificarea e folosită doar ca feedback (cuvinte-cheie), nu ca punctaj.\n"
                "Vezi scorul când finalizezi testul; pentru răspunsul corect bifează „Arată și answer key în aplicație” în Rezultate."
            )

            placeholder = "— alege o strategie —"
            options = [placeholder] + [str(x) for x in allowed if str(x).strip()]
            select_key = f"test_{current.id}_strategy_select"

            if select_key not in st.session_state:
                existing = session.user_answers.get(current.id) or {}
                existing_choice = ""
                if isinstance(existing, dict):
                    existing_choice = str(existing.get("strategy") or existing.get("strategy_label") or "")
                st.session_state[select_key] = existing_choice if existing_choice in options else placeholder

            selected = st.selectbox("Alege strategia:", options=options, key=select_key)
            justification = st.text_area(
                "Justificare (2–3 propoziții):",
                key=answer_key,
                height=120,
            )

            chosen = "" if selected == placeholder else str(selected)
            session.user_answers[current.id] = {
                "strategy": chosen,
                "justification": justification,
                "text": justification,
            }
        elif ui_label == UI_LABEL_ADVERSARIAL:
            st.info("Completează ambele câmpuri: valoarea din rădăcină și numărul de frunze evaluate de Alpha-Beta.")

            val_key = f"test_{current.id}_ab_root_value"
            leaves_key = f"test_{current.id}_ab_visited_leaves"

            if val_key not in st.session_state or leaves_key not in st.session_state:
                existing = session.user_answers.get(current.id) or {}
                if isinstance(existing, dict):
                    st.session_state[val_key] = str(existing.get("root_value") or "")
                    st.session_state[leaves_key] = str(existing.get("visited_leaves") or "")
                else:
                    st.session_state[val_key] = ""
                    st.session_state[leaves_key] = ""

            root_value_text = st.text_input("Valoare în rădăcină:", key=val_key, placeholder="ex: 6")
            visited_leaves_text = st.text_input("Număr frunze evaluate:", key=leaves_key, placeholder="ex: 9")

            session.user_answers[current.id] = {
                "root_value": root_value_text,
                "visited_leaves": visited_leaves_text,
            }
        else:
            user_answer = st.text_area("✍️ Răspunsul tău:", key=answer_key, height=140)
            session.user_answers[current.id] = user_answer

        st.session_state.test_session = session

        col_prev, col_next, col_finish, col_progress = st.columns([1, 1, 1, 2])
        with col_prev:
            if st.button("⬅️ Întrebarea anterioară", disabled=session.current_index <= 0):
                session.current_index -= 1
                st.session_state.test_session = session
                st.rerun()
        with col_next:
            if st.button("Întrebarea următoare ➡️", disabled=session.current_index >= len(session.questions) - 1):
                session.current_index += 1
                st.session_state.test_session = session
                st.rerun()
        with col_finish:
            finish_disabled = len(session.questions) == 0
            if st.button("✅ Finalizează testul", type="primary", disabled=finish_disabled, key="finish_test"):
                st.session_state.test_view = "results"
                st.rerun()
        with col_progress:
            answered = sum(1 for q in session.questions if is_test_answered(q, session.user_answers.get(q.id)))
            st.info(f"Răspunsuri completate: {answered}/{len(session.questions)}")

        if st.session_state.get("test_generation_errors"):
            with st.expander("⚠️ Probleme la generare (detalii)"):
                for err in st.session_state.test_generation_errors:
                    st.write(f"- {err}")

    if debug_mode:
        render_debug_panel()
    st.stop()

if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None

if st.session_state.problem_type != problem_type:
    st.session_state.problem_type = problem_type
    st.session_state.matrix = None
    st.session_state.correct_expl = ""
    st.session_state.user_feedback = ""
    st.session_state.question = None
    st.session_state.test_session = TestSession()
    st.session_state.single_eval_pdf_bytes = None
    
    if problem_type == UI_LABEL_NASH:
        st.session_state.game = NashGame()
    elif problem_type == UI_LABEL_NQUEENS:
        st.session_state.game = NQueensProblem()
    elif problem_type == UI_LABEL_KNIGHTS:
        st.session_state.game = KnightsTourProblem()
    elif problem_type == UI_LABEL_HANOI:
        st.session_state.game = TowerOfHanoiProblem()
    elif problem_type == UI_LABEL_GRAPH_COLORING:
        st.session_state.game = GraphColoringProblem()
    elif problem_type == UI_LABEL_CSP_BT:
        st.session_state.game = None
    elif problem_type == UI_LABEL_ADVERSARIAL:
        st.session_state.game = None
    else:
        st.session_state.game = TowerOfHanoiProblem()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("1. Generare & Export")

    if problem_type == UI_LABEL_CSP_BT:
        st.markdown("### Instanțe CSP predefinite")
        instance_files = list_csp_instances()
        if not instance_files:
            st.warning("Nu am găsit instanțe CSP în `app/data/csp_instances/*.json`.")
        else:
            options = {p.stem: p for p in instance_files}
            selected_name = st.selectbox(
                "Alege instanța:",
                options=list(options.keys()),
                key="single_csp_instance_select",
            )
            selected_path = options[selected_name]

            if st.button("📥 Încarcă instanța", use_container_width=True):
                with st.spinner("Se încarcă instanța și se calculează soluția (BT)..."):
                    try:
                        gen = CSPInstanceProblem(instance_path=selected_path)
                        question = gen.generate_question(
                            ui_label=problem_type,
                            chapter="CSP",
                            extra_metadata={"topic": "CSP (Cerința 3)", "instance_file": selected_path.name},
                        )
                    except Exception as e:
                        st.session_state.matrix = None
                        st.session_state.correct_expl = ""
                        st.session_state.user_feedback = ""
                        st.session_state.question = None
                        st.session_state.test_session = TestSession()
                        st.error(f"Nu s-a putut încărca instanța: {e}")
                    else:
                        if not question.data:
                            st.session_state.matrix = None
                            st.session_state.correct_expl = ""
                            st.session_state.user_feedback = ""
                            st.session_state.question = None
                            st.session_state.test_session = TestSession()
                            st.error(
                                f"Nu s-a putut construi întrebarea: {question.correct_explanation or 'Eroare.'}"
                            )
                        else:
                            st.session_state.question = question
                            st.session_state.test_session = TestSession(questions=[question], current_index=0)
                            st.session_state.matrix = question.data
                            st.session_state.correct_expl = question.correct_explanation
                            st.session_state.user_feedback = ""
                            st.session_state["single_csp_bt_answer"] = ""
                            st.session_state.single_eval_pdf_bytes = None
                            st.success("Instanță încărcată cu succes!")
    elif problem_type == UI_LABEL_ADVERSARIAL:
        st.markdown("### Arbori adversarial (Alpha-Beta)")

        source_mode = st.radio(
            "Sursă arbore:",
            ("Predefinit", "Random"),
            horizontal=True,
            key="single_ab_source_mode",
        )

        if source_mode == "Predefinit":
            tree_files = list_adversarial_trees()
            if not tree_files:
                st.warning("Nu am găsit arbori în `app/data/adversarial_trees/*.json`.")
            else:
                options = {p.stem: p for p in tree_files}
                selected_name = st.selectbox(
                    "Alege arborele:",
                    options=list(options.keys()),
                    key="single_ab_tree_select",
                )
                selected_path = options[selected_name]

                if st.button("📥 Încarcă arborele", use_container_width=True):
                    with st.spinner("Se încarcă arborele și se calculează rezultatul (Alpha-Beta)..."):
                        try:
                            gen = AlphaBetaTreeProblem(instance_path=selected_path)
                            question = gen.generate_question(
                                ui_label=problem_type,
                                chapter="Adversarial",
                                extra_metadata={
                                    "topic": "MinMax + Alpha-Beta (Cerința 4)",
                                    "instance_file": selected_path.name,
                                },
                            )
                        except Exception as e:
                            st.session_state.matrix = None
                            st.session_state.correct_expl = ""
                            st.session_state.user_feedback = ""
                            st.session_state.question = None
                            st.session_state.test_session = TestSession()
                            st.error(f"Nu s-a putut încărca arborele: {e}")
                        else:
                            if not question.data:
                                st.session_state.matrix = None
                                st.session_state.correct_expl = ""
                                st.session_state.user_feedback = ""
                                st.session_state.question = None
                                st.session_state.test_session = TestSession()
                                st.error(
                                    f"Nu s-a putut construi întrebarea: {question.correct_explanation or 'Eroare.'}"
                                )
                            else:
                                st.session_state.question = question
                                st.session_state.test_session = TestSession(questions=[question], current_index=0)
                            st.session_state.matrix = question.data
                            st.session_state.correct_expl = question.correct_explanation
                            st.session_state.user_feedback = ""
                            st.session_state["single_ab_root_value"] = ""
                            st.session_state["single_ab_leaves"] = ""
                            st.session_state.single_eval_pdf_bytes = None
                            st.success("Arbore încărcat cu succes!")
        else:
            depth = int(
                st.slider(
                    "Adâncime (număr de muchii până la frunze):",
                    min_value=1,
                    max_value=4,
                    value=3,
                    step=1,
                    key="single_ab_depth",
                )
            )
            branching = int(
                st.slider(
                    "Factor de ramificare:",
                    min_value=2,
                    max_value=3,
                    value=2,
                    step=1,
                    key="single_ab_branching",
                )
            )

            col_vmin, col_vmax = st.columns(2)
            with col_vmin:
                value_min = int(st.number_input("Valoare minimă frunze:", value=-9, step=1, key="single_ab_vmin"))
            with col_vmax:
                value_max = int(st.number_input("Valoare maximă frunze:", value=9, step=1, key="single_ab_vmax"))

            if st.button("🎲 Generează arbore random", use_container_width=True):
                with st.spinner("Se generează arborele și se calculează rezultatul (Alpha-Beta)..."):
                    try:
                        gen = AlphaBetaTreeProblem(
                            depth=depth,
                            branching=branching,
                            value_min=value_min,
                            value_max=value_max,
                        )
                        question = gen.generate_question(
                            ui_label=problem_type,
                            chapter="Adversarial",
                            extra_metadata={
                                "topic": "MinMax + Alpha-Beta (Cerința 4)",
                                "source": "random",
                            },
                        )
                    except Exception as e:
                        st.session_state.matrix = None
                        st.session_state.correct_expl = ""
                        st.session_state.user_feedback = ""
                        st.session_state.question = None
                        st.session_state.test_session = TestSession()
                        st.error(f"Nu s-a putut genera arborele: {e}")
                    else:
                        if not question.data:
                            st.session_state.matrix = None
                            st.session_state.correct_expl = ""
                            st.session_state.user_feedback = ""
                            st.session_state.question = None
                            st.session_state.test_session = TestSession()
                            st.error(
                                f"Nu s-a putut construi întrebarea: {question.correct_explanation or 'Eroare la generare.'}"
                            )
                        else:
                            prompt_text = build_prompt_text(
                                problem_type,
                                data=question.data,
                                metadata=question.metadata,
                                fallback_game=gen,
                            )
                            if prompt_text:
                                question = replace(question, prompt_text=prompt_text)
                            st.session_state.question = question
                            st.session_state.test_session = TestSession(questions=[question], current_index=0)
                            st.session_state.matrix = question.data
                            st.session_state.correct_expl = question.correct_explanation
                            st.session_state.user_feedback = ""
                            st.session_state["single_ab_root_value"] = ""
                            st.session_state["single_ab_leaves"] = ""
                            st.session_state.single_eval_pdf_bytes = None
                            st.success("Arbore generat cu succes!")
    else:
        if st.button("🎲 Generează Întrebare Nouă", use_container_width=True):
            with st.spinner("Se rulează algoritmul generator..."):
                try:
                    question = st.session_state.game.generate_question(ui_label=problem_type)
                except Exception as e:
                    st.session_state.matrix = None
                    st.session_state.correct_expl = ""
                    st.session_state.user_feedback = ""
                    st.session_state.question = None
                    st.session_state.test_session = TestSession()
                    st.error(f"Nu s-a putut genera problema: {e}")
                else:
                    if not question.data:
                        st.session_state.matrix = None
                        st.session_state.correct_expl = ""
                        st.session_state.user_feedback = ""
                        st.session_state.question = None
                        st.session_state.test_session = TestSession()
                        st.error(
                            f"Nu s-a putut genera problema: {question.correct_explanation or 'Eroare la generare.'}"
                        )
                    else:
                        prompt_text = build_prompt_text(
                            problem_type,
                            data=question.data,
                            metadata=question.metadata,
                            fallback_game=st.session_state.game,
                        )
                        if prompt_text:
                            question = replace(question, prompt_text=prompt_text)
                        st.session_state.question = question
                        st.session_state.test_session = TestSession(questions=[question], current_index=0)
                        st.session_state.matrix = question.data
                        st.session_state.correct_expl = question.correct_explanation
                        st.session_state.user_feedback = ""
                        st.session_state.single_eval_pdf_bytes = None
                        st.success("Problemă generată cu succes!")

    if st.session_state.matrix:
        st.write("---")
        st.write("📄 **Opțiuni Export:**")

        question = st.session_state.get("question")
        
        pdf_req = getattr(question, "prompt_text", "") if question else ""
        if not pdf_req:
            if problem_type == UI_LABEL_NASH:
                pdf_req = "Se da matricea de plati de mai jos. Identificati daca exista un Echilibru Nash pur si specificati coordonatele (ex: L1-C1)."
            elif problem_type == UI_LABEL_NQUEENS:
                board_size = len(st.session_state.matrix)
                pdf_req = f"Pe tabla de {board_size}x{board_size} de mai jos, propuneti o configurare pentru regine astfel incat sa nu se atace reciproc (pe linii, coloane sau diagonale)."
            elif problem_type == UI_LABEL_KNIGHTS:
                board_size = len(st.session_state.matrix)
                pdf_req = f"Pe tabla de {board_size}x{board_size} de mai jos, creati un tur al calului care viziteaza fiecare casuta exact o singura data. Calul se misca in forma de 'L'."
            elif problem_type == UI_LABEL_GRAPH_COLORING:
                metadata = question.metadata if question else {}
                n = int(metadata.get("n") or (len(st.session_state.matrix) if st.session_state.matrix else 0) or 0)
                k = int(metadata.get("k") or len(metadata.get("color_names") or []) or 0)
                colors = metadata.get("color_names") or []
                colors_text = ", ".join(map(str, colors)) if colors else "—"
                pdf_req = (
                    "Se dă un graf neorientat (noduri 1..n) reprezentat prin matricea de adiacență de mai jos. "
                    f"Colorați nodurile folosind cel mult k={k} culori astfel încât două noduri adiacente să nu "
                    f"aibă aceeași culoare. Culori permise: {colors_text}. (n={n})"
                )
            elif problem_type == UI_LABEL_ADVERSARIAL:
                pdf_req = (
                    "Cerința 4 — MinMax + Alpha-Beta: Pentru arborele dat, calculează valoarea din rădăcină și "
                    "numărul de frunze evaluate (vizitate efectiv) de Alpha-Beta, în parcurgere stânga→dreapta."
                )
            else:  # Tower of Hanoi
                num_disks = (question.metadata.get("num_disks") if question else None) or st.session_state.game.num_disks
                num_pegs = (question.metadata.get("num_pegs") if question else None) or st.session_state.game.num_pegs
                peg_names = ["A", "B", "C", "D"][:num_pegs]
                pdf_req = (
                    f"Turnurile din Hanoi: Mutati toate cele {num_disks} discuri de pe tija {peg_names[0]} pe tija {peg_names[-1]}, "
                    "respectand regulile (un disc mai mare nu poate fi plasat peste unul mai mic)."
                )

        try:
            # For Tower of Hanoi, pass the initial state
            if problem_type == UI_LABEL_HANOI:
                hanoi_state = (question.metadata.get("initial_state") if question else None) or st.session_state.game.initial_state
                pdf_bytes = create_pdf(problem_type, pdf_req, st.session_state.matrix, hanoi_state=hanoi_state)
            else:
                pdf_bytes = create_pdf(problem_type, pdf_req, st.session_state.matrix)
            
            st.download_button(
                label="⬇️ Descarcă Subiectul (PDF)",
                data=pdf_bytes,
                file_name="subiect_examen_ia.pdf",
                mime="application/pdf",
                key="dl_single_question_pdf",
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"Nu s-a putut genera PDF-ul: {e}")

        eval_pdf_bytes = st.session_state.get("single_eval_pdf_bytes")
        if eval_pdf_bytes:
            st.download_button(
                label="⬇️ Descarcă Evaluarea (PDF)",
                data=eval_pdf_bytes,
                file_name="evaluare_ia.pdf",
                mime="application/pdf",
                key="dl_single_evaluation_pdf",
                use_container_width=True,
            )

with col_right:
    st.subheader("2. Vizualizare și Răspuns")
    
    if st.session_state.matrix:
        question = st.session_state.get("question")
        if problem_type == UI_LABEL_NASH:
            st.markdown("### Cerință:")
            st.write("Se dă matricea de plăți de mai jos. **Identifică dacă există un Echilibru Nash pur** și specifică coordonatele.")
            
            df_display = pd.DataFrame(
                st.session_state.matrix,
                index=["Linia 1", "Linia 2"],
                columns=["Coloana 1", "Coloana 2"]
            )
            st.table(df_display)
            
        elif problem_type == UI_LABEL_NQUEENS:
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
        
        elif problem_type == UI_LABEL_KNIGHTS:
            # Get the board size and starting position
            board_size = len(st.session_state.matrix)
            start_pos = (question.metadata.get("start_pos") if question else None) or st.session_state.game.start_pos
            
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
        elif problem_type == UI_LABEL_GRAPH_COLORING:
            metadata = question.metadata if question else {}
            n = int(metadata.get("n") or (len(st.session_state.matrix) if st.session_state.matrix else 0) or 0)
            k = int(metadata.get("k") or len(metadata.get("color_names") or []) or 0)
            color_names = list(metadata.get("color_names") or [])
            edges = list(metadata.get("edges") or [])

            st.markdown("### Cerință:")
            if color_names:
                st.write(f"Colorați graful folosind cel mult **{k}** culori: **{', '.join(map(str, color_names))}**.")
            else:
                st.write(f"Colorați graful folosind cel mult **{k}** culori.")

            try:
                df_display = pd.DataFrame(
                    st.session_state.matrix,
                    index=[str(i) for i in range(1, n + 1)] if n else None,
                    columns=[str(i) for i in range(1, n + 1)] if n else None,
                )
                st.dataframe(df_display, use_container_width=True)
            except Exception:
                st.write(st.session_state.matrix)

            with st.expander("Muchii (edge list)"):
                st.write(", ".join(f"({u},{v})" for u, v in edges) if edges else "—")

            input_mode = st.radio(
                "Mod răspuns:",
                ("Dropdown", "Text", "PDF"),
                horizontal=True,
                key="single_graph_coloring_mode",
            )
            if input_mode == "PDF":
                uploaded = st.file_uploader(
                    "📄 Încarcă PDF răspuns:",
                    type=["pdf"],
                    key="single_graph_coloring_pdf",
                )
                pdf_text = ""
                if uploaded is not None:
                    try:
                        pdf_text = extract_text_from_pdf(uploaded.getvalue())
                    except Exception as e:
                        st.error(f"Nu am putut citi PDF-ul: {e}")
                        pdf_text = ""

                graph_assignment = extract_graph_coloring_mapping(pdf_text, n=n, color_names=color_names)
                user_answer = graph_coloring_to_text(graph_assignment) or format_graph_coloring_mapping(graph_assignment)

                if uploaded is not None:
                    with st.expander("🧾 Text extras din PDF"):
                        st.code(pdf_text or "—", language="text")
                if uploaded is not None and not graph_assignment:
                    st.warning("Nu am găsit o mapare nod→culoare în PDF. Exemplu: `1:R, 2:G, 3:B`.")

                if user_answer:
                    with st.expander("📝 Răspuns extras (text)"):
                        st.write(user_answer)

            elif input_mode == "Text":
                user_answer = st.text_area(
                    "✍️ Răspunsul tău (ex: 1:R, 2:G, 3:B):",
                    key="single_graph_coloring_text",
                    height=120,
                )
                parsed = parse_coloring_text(user_answer, n=n, color_names=color_names)
                graph_assignment = parsed.assignment
                if parsed.errors:
                    with st.expander("⚠️ Probleme de parsare"):
                        for err in parsed.errors:
                            st.write(f"- {err}")
            else:
                graph_assignment = render_interactive_graph_coloring(
                    n=n,
                    color_names=color_names,
                    key_prefix="graph_coloring_user",
                )
                user_answer = graph_coloring_to_text(graph_assignment)
                with st.expander("📝 Vezi răspunsul tău (text)"):
                    st.write(user_answer or "Nu ai asignat încă nicio culoare.")

        elif problem_type == UI_LABEL_CSP_BT:
            metadata = question.metadata if question else {}
            csp_meta = metadata.get("csp") or {}
            variables = list(csp_meta.get("variables") or [])
            domains = dict(csp_meta.get("domains") or {})
            partial = dict(metadata.get("partial_assignment") or {})
            remaining_vars = list(metadata.get("remaining_variables") or [v for v in variables if v not in partial])

            solver_options = metadata.get("solver_options") or {}
            method_parts = ["BT"]
            if solver_options.get("mrv"):
                method_parts.append("MRV")
            if solver_options.get("forward_checking"):
                method_parts.append("FC")
            if solver_options.get("ac3_preprocess") or solver_options.get("ac3_interleaved"):
                if solver_options.get("ac3_preprocess") and solver_options.get("ac3_interleaved"):
                    method_parts.append("AC-3(pre+mac)")
                elif solver_options.get("ac3_interleaved"):
                    method_parts.append("AC-3(mac)")
                else:
                    method_parts.append("AC-3(pre)")

            st.markdown("### Cerință:")
            title = metadata.get("csp_instance_title") or metadata.get("csp_instance_id") or ""
            if title:
                st.write(f"Instanță: **{title}**")
            st.write(f"Metodă: **{' + '.join(method_parts)}**")

            try:
                df_display = pd.DataFrame(
                    st.session_state.matrix,
                    columns=["Câmp", "Valoare"],
                )
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            except Exception:
                st.write(st.session_state.matrix)

            if partial:
                partial_text = ", ".join(f"{k}={partial[k]}" for k in variables if k in partial) or "—"
                st.info(f"Asignare parțială: {partial_text}")

            if remaining_vars:
                st.info("Variabile de completat: " + ", ".join(remaining_vars))

            csp_answer_mode = st.radio(
                "Mod răspuns:",
                ("Text", "PDF"),
                horizontal=True,
                key="single_csp_bt_answer_mode",
            )

            if csp_answer_mode == "PDF":
                uploaded = st.file_uploader(
                    "📄 Încarcă PDF răspuns:",
                    type=["pdf"],
                    key="single_csp_bt_pdf",
                )
                pdf_text = ""
                extracted_answer = ""
                if uploaded is not None:
                    try:
                        pdf_text = extract_text_from_pdf(uploaded.getvalue())
                    except Exception as e:
                        st.error(f"Nu am putut citi PDF-ul: {e}")
                        pdf_text = ""

                    pairs = extract_csp_var_value_pairs(pdf_text, allowed_variables=variables)
                    extracted_answer = format_csp_pairs(pairs)
                    st.session_state["single_csp_bt_answer_from_pdf"] = extracted_answer

                    if extracted_answer and not str(st.session_state.get("single_csp_bt_answer") or "").strip():
                        st.session_state["single_csp_bt_answer"] = extracted_answer

                    with st.expander("🧾 Text extras din PDF"):
                        st.code(pdf_text or "—", language="text")

                    if extracted_answer:
                        st.caption(f"Răspuns extras: {extracted_answer}")
                    else:
                        st.warning("Nu am găsit perechi `Var=Val` în PDF (pentru variabilele instanței).")

            user_answer = st.text_area(
                "✍️ Asignarea ta (ex: X=1, Y=2):",
                key="single_csp_bt_answer",
                height=120,
            )

        elif problem_type == UI_LABEL_ADVERSARIAL:
            meta = question.metadata if question else {}
            adv = (meta.get("adversarial") or {}) if isinstance(meta, dict) else {}
            total_leaves = (adv.get("total_leaves") if isinstance(adv, dict) else None) or None

            st.markdown("### Cerință:")
            st.write(
                "Pentru arborele de mai jos, calculează **valoarea din rădăcină** și "
                "**numărul de frunze evaluate (vizitate efectiv)** când rulezi Minimax cu tăieri Alpha-Beta."
            )
            st.caption("Parcurgere: stânga → dreapta (ordinea copiilor din arbore).")
            if total_leaves is not None:
                st.info(f"Număr total frunze în arbore: {total_leaves}")

            try:
                lines: list[str] = []
                for row in (st.session_state.matrix or []):
                    if isinstance(row, (list, tuple)) and row:
                        lines.append(str(row[0]))
                    else:
                        lines.append(str(row))
                st.code("\n".join(lines), language="text")
            except Exception:
                st.write(st.session_state.matrix)

            st.markdown("---")
            st.markdown("### Răspunsul tău")

            ab_answer_mode = st.radio(
                "Mod răspuns:",
                ("Text", "PDF"),
                horizontal=True,
                key="single_ab_answer_mode",
            )

            if ab_answer_mode == "PDF":
                uploaded = st.file_uploader(
                    "📄 Încarcă PDF răspuns:",
                    type=["pdf"],
                    key="single_ab_answer_pdf",
                )
                pdf_text = ""
                if uploaded is not None:
                    try:
                        pdf_text = extract_text_from_pdf(uploaded.getvalue())
                    except Exception as e:
                        st.error(f"Nu am putut citi PDF-ul: {e}")
                        pdf_text = ""

                    value, leaves = extract_minmax_value_and_leaves(pdf_text)
                    if value is not None:
                        st.session_state["single_ab_root_value"] = value
                    if leaves is not None:
                        st.session_state["single_ab_leaves"] = leaves

                    with st.expander("🧾 Text extras din PDF"):
                        st.code(pdf_text or "—", language="text")

                    if value is None and leaves is None:
                        st.warning("Nu am găsit câmpuri `value=...` / `leaves=...` în PDF.")

            _ = st.text_input("Valoare în rădăcină:", key="single_ab_root_value", placeholder="ex: 6")
            _ = st.text_input("Număr frunze evaluate:", key="single_ab_leaves", placeholder="ex: 9")

        else:  # Tower of Hanoi
            num_disks = (question.metadata.get("num_disks") if question else None) or st.session_state.game.num_disks
            num_pegs = (question.metadata.get("num_pegs") if question else None) or st.session_state.game.num_pegs
            initial_state = (question.metadata.get("initial_state") if question else None) or st.session_state.game.initial_state
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
        
        if problem_type == UI_LABEL_NASH:
            nash_answer_mode = st.radio(
                "Mod răspuns:",
                ("Text", "PDF"),
                horizontal=True,
                key="single_nash_answer_mode",
            )

            if nash_answer_mode == "PDF":
                uploaded = st.file_uploader(
                    "📄 Încarcă PDF răspuns:",
                    type=["pdf"],
                    key="single_nash_answer_pdf",
                )
                pdf_text = ""
                if uploaded is not None:
                    try:
                        pdf_text = extract_text_from_pdf(uploaded.getvalue())
                    except Exception as e:
                        st.error(f"Nu am putut citi PDF-ul: {e}")
                        pdf_text = ""

                    coords = extract_nash_coordinates(pdf_text)
                    extracted = ", ".join(coords)
                    st.session_state["single_nash_answer_from_pdf"] = extracted
                    if extracted and not str(st.session_state.get("single_nash_answer") or "").strip():
                        st.session_state["single_nash_answer"] = extracted

                    with st.expander("🧾 Text extras din PDF"):
                        st.code(pdf_text or "—", language="text")

                    if not coords:
                        st.warning("Nu am găsit coordonate de forma `Lx-Cy` în PDF.")

            # Always keep an editable textbox (PDF can pre-fill it).
            user_answer = st.text_area(
                "✍️ Răspunsul tău:",
                key="single_nash_answer",
                height=100,
                placeholder="Scrie explicația aici (sau încarcă un PDF).",
            )
            
            if st.button("✅ Verifică Răspunsul", type="primary"):
                if not user_answer:
                    st.warning("Te rog scrie un răspuns înainte de verificare.")
                else:
                    report_user_answer = user_answer
                    report_feedback = ""

                    if nash_answer_mode == "PDF":
                        expected = set(extract_nash_coordinates(st.session_state.correct_expl))
                        effective_answer = (
                            str(st.session_state.get("single_nash_answer_from_pdf") or "").strip() or user_answer
                        )
                        got = set(extract_nash_coordinates(effective_answer))
                        report_user_answer = effective_answer

                        if expected:
                            correct = expected & got
                            missing = expected - got
                            extra = got - expected

                            score = 100.0 * (len(correct) / max(1, len(expected)))
                            if extra:
                                score = max(0.0, float(score) - 10.0 * len(extra))

                            st.markdown(f"### Scor: **{score:.2f}%**")
                            if score >= 99.99 and not extra:
                                st.success("✅ Corect (coordonate).")
                                report_feedback = "Corect (coordonate)."
                            elif score > 0:
                                st.warning("Parțial (coordonate).")
                                report_feedback = "Partial (coordonate)."
                            else:
                                st.error("❌ Incorect (coordonate).")
                                report_feedback = "Incorect (coordonate)."

                            with st.expander("🔎 Detalii coordonate"):
                                st.write(f"Așteptat: {', '.join(sorted(expected))}")
                                st.write(f"Primit: {', '.join(sorted(got)) or '—'}")
                                if missing:
                                    st.write(f"Lipsește: {', '.join(sorted(missing))}")
                                if extra:
                                    st.write(f"În plus: {', '.join(sorted(extra))}")

                            report_feedback = "\n".join(
                                [
                                    report_feedback,
                                    f"Asteptat: {', '.join(sorted(expected))}",
                                    f"Primit: {', '.join(sorted(got)) or '—'}",
                                    f"Lipseste: {', '.join(sorted(missing))}" if missing else "",
                                    f"In plus: {', '.join(sorted(extra))}" if extra else "",
                                ]
                            ).strip()
                        else:
                            lower = str(effective_answer).lower()
                            ok = ("nu" in lower) and ("echilibru" in lower or "nash" in lower)
                            score = 100.0 if ok else 0.0
                            st.markdown(f"### Scor: **{score:.2f}%**")
                            if ok:
                                st.success("✅ Corect: ai identificat că nu există echilibru Nash pur.")
                                report_feedback = "Corect: ai identificat ca nu exista echilibru Nash pur."
                            else:
                                st.error("❌ Incorect: răspunsul nu indică lipsa unui echilibru Nash pur.")
                                report_feedback = "Incorect: raspunsul nu indica lipsa unui echilibru Nash pur."
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
                        report_feedback = str(feedback)

                    try:
                        st.session_state.single_eval_pdf_bytes = create_evaluation_pdf(
                            question,
                            report_user_answer,
                            float(score),
                            report_feedback,
                            getattr(question, "correct_answer", None) if question else None,
                        )
                    except Exception as e:
                        st.warning(f"Nu s-a putut genera PDF-ul de evaluare: {e}")

                    with st.expander("🔍 Vezi Soluția Algoritmică (Gold Standard)"):
                        st.info(st.session_state.correct_expl)
        elif problem_type == UI_LABEL_NQUEENS:
            # N-Queens verification
            if st.button("✅ Verifică Răspunsul", type="primary"):
                # Get expected number of queens
                expected_queens = (question.metadata.get("expected_queens") if question else None) or st.session_state.game.expected_queens
                
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
                    
                report_score = float(partial_score) if not is_valid else 100.0
                report_feedback = str(validity_msg)
                if detailed_feedback:
                    report_feedback = report_feedback + "\n" + "\n".join(str(x) for x in detailed_feedback)
                if not is_valid:
                    report_feedback = report_feedback + f"\nScor partial: {float(partial_score):.2f}%"

                try:
                    st.session_state.single_eval_pdf_bytes = create_evaluation_pdf(
                        question,
                        user_answer,
                        report_score,
                        report_feedback,
                        getattr(question, "correct_answer", None) if question else None,
                    )
                except Exception as e:
                    st.warning(f"Nu s-a putut genera PDF-ul de evaluare: {e}")

                with st.expander("🔍 Vezi Soluția Algoritmică (Gold Standard)"):
                    st.info(st.session_state.correct_expl)
        
        elif problem_type == UI_LABEL_KNIGHTS:
            # Knight's Tour verification
            if st.button("✅ Verifică Răspunsul", type="primary"):
                # Get starting position and solution
                start_pos = (question.metadata.get("start_pos") if question else None) or st.session_state.game.start_pos
                solution_board = (getattr(question, "correct_answer", None) if question else None) or st.session_state.game.solution_path
                
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

                report_score = float(ai_score) if not is_valid else 100.0
                report_feedback = str(validity_msg)
                if detailed_feedback:
                    report_feedback = report_feedback + "\n" + "\n".join(str(x) for x in detailed_feedback)
                if not is_valid:
                    report_feedback = report_feedback + f"\nScor AI: {float(ai_score):.2f}%"

                try:
                    st.session_state.single_eval_pdf_bytes = create_evaluation_pdf(
                        question,
                        user_answer,
                        report_score,
                        report_feedback,
                        solution_board,
                    )
                except Exception as e:
                    st.warning(f"Nu s-a putut genera PDF-ul de evaluare: {e}")
                
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
        
        elif problem_type == UI_LABEL_GRAPH_COLORING:
            if st.button("✅ Verifică Răspunsul", type="primary"):
                metadata = question.metadata if question else {}
                n = int(metadata.get("n") or (len(st.session_state.matrix) if st.session_state.matrix else 0) or 0)
                color_names = list(metadata.get("color_names") or [])
                edges = list(metadata.get("edges") or [])

                score, msg, details = evaluate_graph_coloring(
                    graph_assignment if "graph_assignment" in locals() else {},
                    n=n,
                    edges=edges,
                    color_names=color_names,
                )

                st.markdown(f"### Scor: **{score:.2f}%**")
                if score >= 99.99:
                    st.success(msg)
                elif score >= 60:
                    st.warning(msg)
                else:
                    st.error(msg)

                if details.get("conflicts"):
                    with st.expander("⚠️ Conflicte (muchii cu aceeași culoare)"):
                        st.write(", ".join(f"{u}-{v}" for u, v in details["conflicts"]))
                if details.get("missing"):
                    with st.expander("⏳ Noduri necolorate"):
                        st.write(", ".join(map(str, details["missing"])))

                report_feedback = str(msg)
                if details.get("conflicts"):
                    report_feedback = report_feedback + "\nConflicte: " + ", ".join(
                        f"{u}-{v}" for u, v in details.get("conflicts") or []
                    )
                if details.get("missing"):
                    report_feedback = report_feedback + "\nNoduri necolorate: " + ", ".join(
                        map(str, details.get("missing") or [])
                    )

                try:
                    st.session_state.single_eval_pdf_bytes = create_evaluation_pdf(
                        question,
                        user_answer if "user_answer" in locals() else "",
                        float(score),
                        report_feedback,
                        getattr(question, "correct_answer", None) if question else None,
                    )
                except Exception as e:
                    st.warning(f"Nu s-a putut genera PDF-ul de evaluare: {e}")

                with st.expander("🔍 Vezi Soluția (Gold Standard)"):
                    st.info(st.session_state.correct_expl)
                    if question and getattr(question, "correct_answer", None):
                        st.write(question.correct_answer)

        elif problem_type == UI_LABEL_CSP_BT:
            if st.button("✅ Verifică Răspunsul", type="primary"):
                metadata = question.metadata if question else {}
                csp_meta = metadata.get("csp") or {}
                variables = list(csp_meta.get("variables") or [])
                domains = dict(csp_meta.get("domains") or {})
                partial = dict(metadata.get("partial_assignment") or {})

                answer_text = str(st.session_state.get("single_csp_bt_answer") or "")
                if st.session_state.get("single_csp_bt_answer_mode") == "PDF":
                    answer_text = str(st.session_state.get("single_csp_bt_answer_from_pdf") or answer_text)

                score, msg, details = evaluate_csp_backtracking_answer(
                    answer_text,
                    variables=variables,
                    domains=domains,
                    partial_assignment=partial,
                    expected_solution=getattr(question, "correct_answer", None) if question else None,
                )

                st.markdown(f"### Scor: **{score:.2f}%**")
                if score >= 99.99:
                    st.success(msg)
                elif score >= 60:
                    st.warning(msg)
                else:
                    st.error(msg)

                if details.get("parse_errors"):
                    with st.expander("⚠️ Probleme de parsare"):
                        for err in details["parse_errors"]:
                            st.write(f"- {err}")

                if details.get("missing"):
                    with st.expander("⏳ Variabile lipsă"):
                        st.write(", ".join(map(str, details["missing"])))

                if details.get("wrong"):
                    with st.expander("❌ Variabile greșite (așteptat vs. primit)"):
                        for var, info in details["wrong"].items():
                            st.write(f"- {var}: așteptat `{info.get('expected')}`, primit `{info.get('got')}`")

                report_feedback = str(msg)
                if details.get("parse_errors"):
                    report_feedback = report_feedback + "\nProbleme parsare:\n- " + "\n- ".join(
                        str(e) for e in (details.get("parse_errors") or [])
                    )
                if details.get("missing"):
                    report_feedback = report_feedback + "\nVariabile lipsa: " + ", ".join(
                        map(str, details.get("missing") or [])
                    )
                if details.get("wrong"):
                    wrong_lines = []
                    for var, info in (details.get("wrong") or {}).items():
                        wrong_lines.append(f"{var}: asteptat {info.get('expected')}, primit {info.get('got')}")
                    if wrong_lines:
                        report_feedback = report_feedback + "\nVariabile gresite:\n- " + "\n- ".join(wrong_lines)

                try:
                    st.session_state.single_eval_pdf_bytes = create_evaluation_pdf(
                        question,
                        answer_text,
                        float(score),
                        report_feedback,
                        getattr(question, "correct_answer", None) if question else None,
                    )
                except Exception as e:
                    st.warning(f"Nu s-a putut genera PDF-ul de evaluare: {e}")

                with st.expander("🔍 Vezi Soluția (Gold Standard)"):
                    st.info(st.session_state.correct_expl)
                    if question and getattr(question, "correct_answer", None):
                        st.write(question.correct_answer)

        elif problem_type == UI_LABEL_ADVERSARIAL:
            if st.button("✅ Verifică Răspunsul", type="primary"):
                expected = getattr(question, "correct_answer", None) if question else None
                score, msg, details = evaluate_alpha_beta_answer(
                    st.session_state.get("single_ab_root_value", ""),
                    st.session_state.get("single_ab_leaves", ""),
                    expected=expected if isinstance(expected, dict) else None,
                )

                st.markdown(f"### Scor: **{score:.2f}%**")
                if score >= 99.99:
                    st.success(msg)
                elif score >= 50:
                    st.warning(msg)
                else:
                    st.error(msg)

                if details.get("errors"):
                    with st.expander("⚠️ Probleme de parsare"):
                        for err in details["errors"]:
                            st.write(f"- {err}")

                user_answer_text = (
                    f"value={st.session_state.get('single_ab_root_value', '')} "
                    f"leaves={st.session_state.get('single_ab_leaves', '')}"
                ).strip()
                report_feedback = str(msg)
                if details.get("errors"):
                    report_feedback = report_feedback + "\nProbleme parsare:\n- " + "\n- ".join(
                        str(e) for e in (details.get("errors") or [])
                    )

                try:
                    st.session_state.single_eval_pdf_bytes = create_evaluation_pdf(
                        question,
                        user_answer_text,
                        float(score),
                        report_feedback,
                        expected,
                    )
                except Exception as e:
                    st.warning(f"Nu s-a putut genera PDF-ul de evaluare: {e}")

                with st.expander("🔍 Vezi Soluția (Gold Standard)"):
                    st.info(st.session_state.correct_expl)
                    if question and getattr(question, "correct_answer", None):
                        st.write(question.correct_answer)

        else:  # Tower of Hanoi verification
            if st.button("✅ Verifică Răspunsul", type="primary"):
                num_disks = (question.metadata.get("num_disks") if question else None) or st.session_state.game.num_disks
                num_pegs = (question.metadata.get("num_pegs") if question else None) or st.session_state.game.num_pegs
                target_peg = num_pegs - 1
                solution_moves = (getattr(question, "correct_answer", None) if question else None) or st.session_state.game.solution_moves
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
                
                report_score = 0.0
                if is_complete and is_optimal:
                    report_score = 100.0
                elif is_complete and not is_optimal:
                    report_score = float(score)

                report_feedback = str(validity_msg)
                if detailed_feedback:
                    report_feedback = report_feedback + "\n" + "\n".join(str(x) for x in detailed_feedback)

                try:
                    st.session_state.single_eval_pdf_bytes = create_evaluation_pdf(
                        question,
                        user_answer,
                        report_score,
                        report_feedback,
                        solution_moves,
                    )
                except Exception as e:
                    st.warning(f"Nu s-a putut genera PDF-ul de evaluare: {e}")

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

if debug_mode:
    render_debug_panel()
