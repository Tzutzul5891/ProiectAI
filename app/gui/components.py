import streamlit as st

def render_interactive_queens_board(n=4, key_prefix="queens"):
    """
    Renders an interactive n-queens board where users can click to place/remove queens.
    Returns the board state as a list of lists.
    """
    # Initialize session state for the board if not exists
    board_key = f"{key_prefix}_board"
    size_key = f"{key_prefix}_size"
    
    # Reset board if size changed
    if size_key not in st.session_state or st.session_state[size_key] != n:
        st.session_state[board_key] = [[False for _ in range(n)] for _ in range(n)]
        st.session_state[size_key] = n
    elif board_key not in st.session_state:
        st.session_state[board_key] = [[False for _ in range(n)] for _ in range(n)]
    
    st.markdown("### 🎯 Tabla Interactivă - Plasează Reginele")
    st.markdown("**Instrucțiuni:** Click pe căsuțe pentru a plasa/șterge regine (♛)")
    
    # Create the interactive board using columns
    for row in range(n):
        cols = st.columns(n)
        for col in range(n):
            with cols[col]:
                # Determine button appearance
                is_queen = st.session_state[board_key][row][col]
                button_label = "♛" if is_queen else "⬜"
                button_type = "primary" if is_queen else "secondary"
                
                # Create button with callback
                if st.button(
                    button_label,
                    key=f"{key_prefix}_btn_{row}_{col}",
                    type=button_type,
                    use_container_width=True
                ):
                    # Toggle queen placement
                    st.session_state[board_key][row][col] = not st.session_state[board_key][row][col]
                    st.rerun()
    
    # Add a reset button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔄 Resetează Tabla", key=f"{key_prefix}_reset"):
            st.session_state[board_key] = [[False for _ in range(n)] for _ in range(n)]
            st.rerun()
    
    with col2:
        # Count queens placed (don't reveal expected count)
        queen_count = sum(sum(row) for row in st.session_state[board_key])
        st.info(f"👑 {queen_count} regine plasate")
    
    return st.session_state[board_key]


def board_to_text(board):
    """
    Converts a board state (list of lists of booleans) to text format.
    Returns a string describing queen positions.
    """
    positions = []
    n = len(board)
    
    for row in range(n):
        for col in range(n):
            if board[row][col]:
                positions.append(f"Rând {row+1}, Col {col+1}")
    
    if not positions:
        return "Nu au fost plasate regine pe tablă."
    
    text = f"Am plasat {len(positions)} regine la următoarele coordonate: {'; '.join(positions)}. "
    
    # Add explanation about the placement
    if len(positions) == n:
        text += "Această configurare reprezintă o încercare de rezolvare a problemei N-Regine, unde nicio regină nu ar trebui să se atace reciproc pe linii, coloane sau diagonale."
    
    return text


def check_queens_validity(board, expected_queens=None):
    """
    Checks if the queens placement is valid (no two queens attack each other).
    Returns (is_valid, message, detailed_feedback).
    """
    n = len(board)
    queens = []
    
    # Find all queen positions
    for row in range(n):
        for col in range(n):
            if board[row][col]:
                queens.append((row, col))
    
    detailed_feedback = []
    errors = []
    
    # Check if we have the expected number of queens
    if expected_queens is not None:
        if len(queens) != expected_queens:
            errors.append(f"Număr incorect de regine: ai plasat {len(queens)}, dar tabla are dimensiunea {n}x{n} și necesită {expected_queens} regine.")
            detailed_feedback.append(f"📊 Numărul corect de regine pentru o tablă {n}x{n} este {expected_queens} (câte una pe fiecare rând/coloană).")
    
    # Check for conflicts
    row_conflicts = {}
    col_conflicts = {}
    diag_conflicts = []
    
    for i in range(len(queens)):
        r1, c1 = queens[i]
        
        # Check same row
        if r1 not in row_conflicts:
            row_conflicts[r1] = []
        row_conflicts[r1].append((r1, c1))
        
        # Check same column
        if c1 not in col_conflicts:
            col_conflicts[c1] = []
        col_conflicts[c1].append((r1, c1))
        
        for j in range(i + 1, len(queens)):
            r2, c2 = queens[j]
            
            # Same diagonal
            if abs(r1 - r2) == abs(c1 - c2):
                diag_conflicts.append(((r1, c1), (r2, c2)))
    
    # Report row conflicts
    for row, positions in row_conflicts.items():
        if len(positions) > 1:
            pos_str = ", ".join([f"Col {c+1}" for r, c in positions])
            errors.append(f"Conflict pe rândul {row+1}: regine la {pos_str}")
            detailed_feedback.append(f"❌ Rândul {row+1}: Două sau mai multe regine pe același rând se atacă între ele.")
    
    # Report column conflicts
    for col, positions in col_conflicts.items():
        if len(positions) > 1:
            pos_str = ", ".join([f"Rând {r+1}" for r, c in positions])
            errors.append(f"Conflict pe coloana {col+1}: regine la {pos_str}")
            detailed_feedback.append(f"❌ Coloana {col+1}: Două sau mai multe regine pe aceeași coloană se atacă între ele.")
    
    # Report diagonal conflicts
    for (r1, c1), (r2, c2) in diag_conflicts:
        errors.append(f"Conflict diagonal: Regina la Rând {r1+1}, Col {c1+1} atacă regina de la Rând {r2+1}, Col {c2+1}")
        detailed_feedback.append(f"❌ Atac diagonal: Rând {r1+1}, Col {c1+1} ↔ Rând {r2+1}, Col {c2+1}")
    
    if not errors:
        return True, "✅ Configurarea este validă! Nicio regină nu se atacă reciproc.", ["✨ Perfect! Toate reginele sunt plasate corect."]
    else:
        summary = f"Configurația are {len(errors)} eroare/erori:"
        return False, summary, detailed_feedback


def render_interactive_knights_board(n=5, start_pos=(0, 0), key_prefix="knights"):
    """
    Renders an interactive Knight's Tour board where users can click to create a path.
    Returns the board state with move numbers.
    """
    board_key = f"{key_prefix}_board"
    size_key = f"{key_prefix}_size"
    move_counter_key = f"{key_prefix}_move_counter"
    
    # Reset board if size changed or doesn't exist
    if (size_key not in st.session_state or st.session_state[size_key] != n or 
        board_key not in st.session_state):
        st.session_state[board_key] = [[-1 for _ in range(n)] for _ in range(n)]
        st.session_state[size_key] = n
        st.session_state[move_counter_key] = 0
        # Set starting position
        st.session_state[board_key][start_pos[0]][start_pos[1]] = 0
    
    st.markdown("### ♞ Tabla Interactivă - Turul Calului")
    st.markdown(f"**Instrucțiuni:** Click pe căsuțe pentru a marca traseul calului. Începe de la poziția marcată cu **0** (Rând {start_pos[0]+1}, Col {start_pos[1]+1})")
    st.info("🐴 Calul se mișcă în formă de 'L': 2 căsuțe într-o direcție + 1 căsuță perpendicular")
    
    # Create the interactive board using columns
    for row in range(n):
        cols = st.columns(n)
        for col in range(n):
            with cols[col]:
                cell_value = st.session_state[board_key][row][col]
                
                # Determine button appearance
                if cell_value == -1:
                    button_label = "⬜"
                    button_type = "secondary"
                else:
                    button_label = str(cell_value)
                    button_type = "primary"
                
                # Create button
                if st.button(
                    button_label,
                    key=f"{key_prefix}_btn_{row}_{col}",
                    type=button_type,
                    use_container_width=True
                ):
                    if cell_value == -1:
                        # Add move
                        st.session_state[move_counter_key] += 1
                        st.session_state[board_key][row][col] = st.session_state[move_counter_key]
                    else:
                        # Remove move (and all subsequent moves)
                        remove_from = cell_value
                        for r in range(n):
                            for c in range(n):
                                if st.session_state[board_key][r][c] >= remove_from:
                                    st.session_state[board_key][r][c] = -1
                        st.session_state[move_counter_key] = remove_from - 1
                    st.rerun()
    
    # Add controls
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔄 Resetează Tabla", key=f"{key_prefix}_reset"):
            st.session_state[board_key] = [[-1 for _ in range(n)] for _ in range(n)]
            st.session_state[board_key][start_pos[0]][start_pos[1]] = 0
            st.session_state[move_counter_key] = 0
            st.rerun()
    
    with col2:
        moves_made = st.session_state[move_counter_key]
        total_squares = n * n
        if moves_made == total_squares - 1:
            st.success(f"✓ {moves_made}/{total_squares-1} mișcări")
        else:
            st.info(f"♞ {moves_made}/{total_squares-1} mișcări")
    
    return st.session_state[board_key]


def knights_board_to_text(board, start_pos):
    """Convert Knight's Tour board to text description"""
    n = len(board)
    moves = []
    
    # Collect all moves in order
    move_positions = {}
    for row in range(n):
        for col in range(n):
            if board[row][col] >= 0:
                move_positions[board[row][col]] = (row, col)
    
    if not move_positions:
        return "Nu au fost marcate mișcări pe tablă."
    
    # Create path description
    sorted_moves = sorted(move_positions.items())
    path_desc = f"Traseul calului începe de la Rând {start_pos[0]+1}, Col {start_pos[1]+1} și continuă prin: "
    
    moves_list = []
    for move_num, (row, col) in sorted_moves[1:]:  # Skip starting position
        moves_list.append(f"Rând {row+1}, Col {col+1}")
    
    if moves_list:
        path_desc += "; ".join(moves_list[:5])  # Show first 5 moves
        if len(moves_list) > 5:
            path_desc += f"; ... și continuă prin {len(moves_list) - 5} mișcări suplimentare"
    
    path_desc += f". Total: {len(sorted_moves)} căsuțe vizitate din {n*n}."
    
    return path_desc


def compute_knights_tour_score(board, solution_board, start_pos):
    """
    AI-based scoring using graph similarity and heuristics.
    Uses multiple metrics to grade the Knight's Tour attempt.
    """
    n = len(board)
    total_squares = n * n
    
    # Collect user's moves
    user_positions = {}
    for row in range(n):
        for col in range(n):
            if board[row][col] >= 0:
                user_positions[board[row][col]] = (row, col)
    
    # Collect solution moves
    solution_positions = {}
    for row in range(n):
        for col in range(n):
            if solution_board[row][col] >= 0:
                solution_positions[solution_board[row][col]] = (row, col)
    
    # Metric 1: Coverage Score (how many squares visited)
    coverage_score = len(user_positions) / total_squares
    
    # Metric 2: Valid Move Sequence Score
    sorted_user_moves = sorted(user_positions.items())
    valid_moves = 0
    total_moves = len(sorted_user_moves) - 1
    
    for i in range(len(sorted_user_moves) - 1):
        _, (r1, c1) = sorted_user_moves[i]
        _, (r2, c2) = sorted_user_moves[i + 1]
        dx, dy = abs(r2 - r1), abs(c2 - c1)
        if (dx == 2 and dy == 1) or (dx == 1 and dy == 2):
            valid_moves += 1
    
    move_validity_score = valid_moves / total_moves if total_moves > 0 else 0
    
    # Metric 3: Path Similarity (edit distance between paths)
    path_similarity = 0
    if len(user_positions) > 0:
        matching_positions = 0
        for move_num in range(min(len(user_positions), len(solution_positions))):
            if move_num in user_positions and move_num in solution_positions:
                user_pos = user_positions[move_num]
                solution_pos = solution_positions[move_num]
                # Manhattan distance between positions
                distance = abs(user_pos[0] - solution_pos[0]) + abs(user_pos[1] - solution_pos[1])
                # Closer positions get higher scores
                matching_positions += max(0, 1 - distance / (2 * n))
        
        path_similarity = matching_positions / min(len(user_positions), len(solution_positions))
    
    # Metric 4: Warnsdorff Heuristic Alignment
    # Check if user followed good heuristics (moving to squares with fewer onward moves)
    heuristic_score = 0
    if len(sorted_user_moves) > 1:
        good_moves = 0
        for i in range(min(5, len(sorted_user_moves) - 1)):  # Check first 5 moves
            _, (r, c) = sorted_user_moves[i]
            # Check if this was a strategic position (corner/edge squares first)
            if r == 0 or r == n-1 or c == 0 or c == n-1:
                good_moves += 1
        heuristic_score = good_moves / min(5, len(sorted_user_moves) - 1)
    
    # Metric 5: Graph Connectivity (are moves forming a connected path?)
    connectivity_score = 1.0 if move_validity_score == 1.0 else move_validity_score * 0.8
    
    # Weighted combination of metrics
    final_score = (
        coverage_score * 0.25 +        # 25% - coverage
        move_validity_score * 0.40 +   # 40% - valid moves
        path_similarity * 0.20 +       # 20% - similarity to solution
        heuristic_score * 0.05 +       # 5% - strategic thinking
        connectivity_score * 0.10      # 10% - path connectivity
    ) * 100
    
    return min(final_score, 100), {
        'coverage': coverage_score,
        'validity': move_validity_score,
        'similarity': path_similarity,
        'heuristic': heuristic_score,
        'connectivity': connectivity_score,
        'valid_moves': valid_moves,
        'total_moves': total_moves
    }


def check_knights_tour_validity(board, start_pos):
    """
    Check if the Knight's Tour path is valid.
    Returns (is_valid, message, detailed_feedback).
    """
    n = len(board)
    total_squares = n * n
    
    # Knight's possible moves
    move_x = [2, 1, -1, -2, -2, -1, 1, 2]
    move_y = [1, 2, 2, 1, -1, -2, -2, -1]
    
    # Collect moves
    move_positions = {}
    for row in range(n):
        for col in range(n):
            if board[row][col] >= 0:
                move_positions[board[row][col]] = (row, col)
    
    errors = []
    detailed_feedback = []
    
    # Check if all squares visited
    if len(move_positions) != total_squares:
        errors.append(f"Turul incomplet: {len(move_positions)}/{total_squares} căsuțe vizitate")
        detailed_feedback.append(f"📊 Trebuie să vizitezi toate cele {total_squares} căsuțe ale tablei {n}x{n}.")
    
    # Check if path is valid (each move follows knight's L-shape)
    sorted_moves = sorted(move_positions.items())
    invalid_moves_count = 0
    
    for i in range(len(sorted_moves) - 1):
        move_num, (r1, c1) = sorted_moves[i]
        next_num, (r2, c2) = sorted_moves[i + 1]
        
        # Check if it's a valid knight move
        dx = abs(r2 - r1)
        dy = abs(c2 - c1)
        
        if not ((dx == 2 and dy == 1) or (dx == 1 and dy == 2)):
            invalid_moves_count += 1
            errors.append(f"Mișcare invalidă: de la Rând {r1+1}, Col {c1+1} la Rând {r2+1}, Col {c2+1}")
            detailed_feedback.append(f"❌ Mișcarea {move_num} → {next_num}: Calul nu se poate mișca de la Rând {r1+1}, Col {c1+1} la Rând {r2+1}, Col {c2+1}. Calul se mișcă în formă de 'L'!")
    
    if not errors:
        return True, "✅ Turul calului este valid! Toate mișcările sunt corecte.", ["✨ Perfect! Ai completat un tur valid al calului!"], 0
    else:
        summary = f"Turul are {len(errors)} eroare/erori:"
        return False, summary, detailed_feedback, invalid_moves_count


def render_interactive_hanoi(num_disks, num_pegs, initial_state, key_prefix="hanoi"):
    """
    Renders an interactive Tower of Hanoi puzzle.
    Returns the move history.
    """
    pegs_key = f"{key_prefix}_pegs"
    moves_key = f"{key_prefix}_moves"
    selected_key = f"{key_prefix}_selected"
    
    peg_names = ["A", "B", "C", "D"][:num_pegs]
    
    # Initialize state
    if pegs_key not in st.session_state:
        st.session_state[pegs_key] = {i: initial_state[i].copy() for i in range(num_pegs)}
        st.session_state[moves_key] = []
        st.session_state[selected_key] = None
    
    st.markdown("### 🗼 Turnurile din Hanoi - Interactiv")
    st.markdown(f"**Instrucțiuni:** Mută discurile de pe tija **{peg_names[0]}** pe tija **{peg_names[-1]}**")
    st.info("📏 **Reguli:** (1) Doar un disc odată. (2) Un disc mare nu poate fi peste unul mic.")
    
    # Display pegs
    cols = st.columns(num_pegs)
    
    for peg_idx in range(num_pegs):
        with cols[peg_idx]:
            st.markdown(f"**Tija {peg_names[peg_idx]}**")
            
            # Show disks on this peg (top to bottom)
            disks = st.session_state[pegs_key][peg_idx]
            
            if disks:
                for disk in disks:
                    disk_label = "🟦" * disk
                    st.text(disk_label)
            else:
                st.text("│")
                st.text("│")
            
            # Button to select/move
            if st.session_state[selected_key] is None:
                # Select mode
                if disks:
                    if st.button(f"📤 Ridică disc", key=f"{key_prefix}_pick_{peg_idx}"):
                        st.session_state[selected_key] = peg_idx
                        st.rerun()
            else:
                # Place mode
                if st.session_state[selected_key] != peg_idx:
                    source_peg = st.session_state[selected_key]
                    source_disks = st.session_state[pegs_key][source_peg]
                    target_disks = st.session_state[pegs_key][peg_idx]
                    
                    # Check if move is valid
                    can_place = len(target_disks) == 0 or source_disks[-1] < target_disks[-1]
                    
                    button_label = f"📥 Plasează" if can_place else "❌ Invalid"
                    button_type = "primary" if can_place else "secondary"
                    
                    if st.button(button_label, key=f"{key_prefix}_place_{peg_idx}", type=button_type, disabled=not can_place):
                        # Move disk
                        disk = st.session_state[pegs_key][source_peg].pop()
                        st.session_state[pegs_key][peg_idx].append(disk)
                        st.session_state[moves_key].append((source_peg, peg_idx))
                        st.session_state[selected_key] = None
                        st.rerun()
                else:
                    if st.button(f"↩️ Anulează", key=f"{key_prefix}_cancel_{peg_idx}"):
                        st.session_state[selected_key] = None
                        st.rerun()
    
    # Controls
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔄 Resetează", key=f"{key_prefix}_reset"):
            st.session_state[pegs_key] = {i: initial_state[i].copy() for i in range(num_pegs)}
            st.session_state[moves_key] = []
            st.session_state[selected_key] = None
            st.rerun()
    
    with col2:
        moves_count = len(st.session_state[moves_key])
        st.info(f"🔢 {moves_count} mișcări")
    
    return st.session_state[moves_key], st.session_state[pegs_key]


def hanoi_moves_to_text(moves, num_pegs):
    """Convert Hanoi moves to text description"""
    peg_names = ["A", "B", "C", "D"][:num_pegs]
    
    if not moves:
        return "Nu au fost efectuate mișcări încă."
    
    move_descriptions = [f"{peg_names[src]} → {peg_names[dst]}" for src, dst in moves]
    
    text = f"Ai efectuat {len(moves)} mișcări: "
    if len(moves) <= 10:
        text += ", ".join(move_descriptions)
    else:
        text += ", ".join(move_descriptions[:5]) + f"... și încă {len(moves) - 5} mișcări"
    
    return text


def check_hanoi_validity(moves, pegs_state, num_disks, num_pegs, target_peg, optimal_solution_length):
    """
    Check if the Tower of Hanoi solution is valid.
    Returns (is_complete, is_optimal, message, detailed_feedback, efficiency_score).
    """
    peg_names = ["A", "B", "C", "D"][:num_pegs]
    errors = []
    detailed_feedback = []
    
    # Check if puzzle is complete
    target_disks = pegs_state[target_peg]
    is_complete = len(target_disks) == num_disks and target_disks == list(range(num_disks, 0, -1))
    
    if not is_complete:
        errors.append(f"Puzzle incomplet: Nu toate discurile sunt pe tija {peg_names[target_peg]}")
        detailed_feedback.append(f"📊 Trebuie să muți toate cele {num_disks} discuri pe tija {peg_names[target_peg]}.")
    
    # Use the actual optimal solution length from the generated problem
    optimal_moves = optimal_solution_length
    
    # Check efficiency
    user_moves = len(moves)
    efficiency = min(1.0, optimal_moves / user_moves) if user_moves > 0 else 0
    
    if is_complete:
        if user_moves <= optimal_moves:
            return True, True, "✅ Perfect! Ai rezolvat puzzle-ul cu numărul minim de mișcări!", ["✨ Soluție optimă!"], 1.0
        elif user_moves <= optimal_moves * 1.5:
            detailed_feedback.append(f"✅ Ai completat puzzle-ul!")
            detailed_feedback.append(f"📊 Ai folosit {user_moves} mișcări. Soluția optimă: {optimal_moves} mișcări.")
            detailed_feedback.append(f"💡 Eficiență: {efficiency*100:.1f}%")
            return True, False, f"✅ Completat! ({user_moves} mișcări vs {optimal_moves} optime)", detailed_feedback, efficiency
        else:
            detailed_feedback.append(f"✅ Ai completat puzzle-ul, dar cu multe mișcări în plus.")
            detailed_feedback.append(f"📊 Ai folosit {user_moves} mișcări. Soluția optimă: {optimal_moves} mișcări.")
            detailed_feedback.append(f"💡 Încearcă să planifici mișcările mai eficient!")
            return True, False, f"✅ Completat, dar neoptimal ({user_moves}/{optimal_moves})", detailed_feedback, efficiency
    else:
        detailed_feedback.append(f"📊 Ai efectuat {user_moves} mișcări până acum.")
        detailed_feedback.append(f"🎯 Continuă să muți discurile pe tija {peg_names[target_peg]}.")
        return False, False, "⏳ Puzzle incomplet", detailed_feedback, 0
