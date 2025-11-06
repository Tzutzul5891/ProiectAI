#!/usr/bin/env python3

"""
A terminal-based quiz application that uses a Knowledge Graph
and simulated AI calls to generate questions and grade answers
on computer science topics.
"""

import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from llama_cpp import Llama

try:
    from llama_cpp import Llama as LlamaClient  # type: ignore
except ImportError:  # pragma: no cover
    LlamaClient = None  # type: ignore
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
LLAMA_INSTANCE: Optional["Llama"] = None  # type: ignore[name-defined]

# --- LOCAL MODEL MANAGEMENT ---
def ensure_model() -> None:
    """
    Make sure there is at least one GGUF file under models/.
    If none exist, run the downloader script defined in scripts/get_model.py.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if any(MODELS_DIR.glob("*.gguf")):
        return

    downloader = ROOT / "scripts" / "get_model.py"
    if not downloader.exists():
        raise FileNotFoundError("Model downloader script missing at scripts/get_model.py.")

    print("No local model detected. Downloading according to model.lock ...")
    subprocess.check_call([sys.executable, str(downloader)])


def get_model_path() -> Path:
    files = sorted(MODELS_DIR.glob("*.gguf"))
    if not files:
        raise FileNotFoundError("No GGUF model present under models/.")
    return files[0]


def init_llama() -> Optional["Llama"]:
    if LlamaClient is None:
        print("[LLM Warning] llama-cpp-python is not available; falling back to template questions.")
        return None
    try:
        ensure_model()
        model_path = get_model_path()
        print(f"[LLM] Loading local model from {model_path} ...")
        return LlamaClient(model_path=str(model_path), n_ctx=2048, n_threads=4)
    except Exception as exc:
        print(f"[LLM Warning] Failed to initialize local model: {exc}")
        return None


def get_llama() -> Optional["Llama"]:
    global LLAMA_INSTANCE
    if LLAMA_INSTANCE is not None:
        return LLAMA_INSTANCE
    LLAMA_INSTANCE = init_llama()
    return LLAMA_INSTANCE

# --- 1. THE KNOWLEDGE GRAPH (KG) ---
# Defines the core problems and their factual properties.
# We are returning to this static graph for reliable partial credit.
KNOWLEDGE_GRAPH: Dict[str, Any] = {
    "N-Queens": {
        "Type": "Constraint Satisfaction Problem",
        "Solutions": ["Backtracking"],
        "Complexity": "O(N!)",
    },
    "Generalized Hanoi": {
        "Type": "Recursion",
        "Solutions": ["Divide and Conquer"],
        "Complexity": "O(2^N)",
    },
    "Graph Coloring": {
        "Type": "NP-Complete Problem",
        "Solutions": ["Backtracking", "Greedy Algorithm"],
        "Complexity": "NP-Complete",
    },
    "Knight’s Tour": {
        "Type": "Hamiltonian Path Problem",
        "Solutions": ["Backtracking", "Warnsdorff's Rule"],
        "Complexity": "Varies (Heuristic/Exponential)",
    },
}

# --- 2. WEIGHTED SEMANTIC LINKS ---
# This graph defines the "closeness" between concepts for partial credit.
# The score (0.0 to 1.0) is the partial credit awarded.
SEMANTIC_LINKS: Dict[str, Dict[str, float]] = {
    # --- Solution Concepts ---
    "Backtracking": {
        "DFS": 0.9,  # Backtracking is a form of DFS
        "Search Algorithm": 0.7,  # "is_a" (parent category)
        "Constraint Programming": 0.6,  # Related technique
        "BFS": 0.4,  # Different type of search
        "Heuristic Search": 0.5,  # Different type of search
        "Recursion": 0.7,  # Often implemented recursively
        "Recursive Algorithm": 0.7  # Often implemented recursively
    },
    "DFS": {
        "Backtracking": 0.9,
        "Search Algorithm": 0.7,
        "BFS": 0.4,
        "Graph Theory Problem": 0.6,  # Fundamental graph algorithm
        "Recursive Algorithm": 0.8  # Often implemented recursively
    },
    "BFS": {
        "DFS": 0.4,
        "Search Algorithm": 0.7,
        "Graph Theory Problem": 0.6,  # Fundamental graph algorithm
        "A* Search": 0.6,  # A* is a related best-first search
        "Iterative Algorithm": 0.8  # Often implemented iteratively
    },
    "Heuristic Search": {
        "Search Algorithm": 0.7,
        "A* Search": 0.9,
        "Warnsdorff's Rule": 0.8,
        "Backtracking": 0.5,
        "Heuristic Algorithm": 0.9  # Very closely related
    },
    "Warnsdorff's Rule": {
        "Heuristic Algorithm": 0.8,
        "Heuristic Search": 0.8,
        "Greedy Algorithm": 0.6  # It's a greedy heuristic
    },
    "Greedy Algorithm": {
        "Approximation Algorithm": 0.8,
        "Heuristic Algorithm": 0.7,
        "Warnsdorff's Rule": 0.6,
        "Optimization Technique": 0.8  # It's an optimization technique
    },
    "Recursive Algorithm": {
        "Recursion": 1.0,  # Synonym
        "Iterative Algorithm": 0.3,  # Opposite
        "Divide and Conquer": 0.7,  # Often implemented with recursion
        "Backtracking": 0.7,
        "DFS": 0.8
    },
    "Divide and Conquer": {
        "Recursive Algorithm": 0.7,
        "Algorithmic Paradigm": 0.8
    },
    "Constraint Programming": {
        "Constraint Satisfaction Problem": 0.8,
        "Backtracking": 0.6
    },

    # --- Type Concepts ---
    "Constraint Satisfaction Problem": {
        "Search Problem": 0.8,  # Parent
        "Algorithmic Paradigm": 0.5,  # Grandparent
        "NP-Complete Problem": 0.3,  # Often related, but not the same
        "Constraint Programming": 0.8,
        "Graph Coloring": 0.5  # A classic example
    },
    "Hamiltonian Path Problem": {
        "Graph Theory Problem": 0.8,  # Parent
        "NP-Complete Problem": 0.7,  # HPP is NP-Complete
        "Discrete Mathematics": 0.5  # Grandparent
    },
    "NP-Complete Problem": {
        "Hamiltonian Path Problem": 0.7,
        "Graph Coloring": 0.7,
        "Computational Complexity Class": 0.8,  # Parent
        "Theoretical Computer Science": 0.5,  # Grandparent
        "Search Problem": 0.6  # Many are search problems
    },
    "Recursion": {
        "Recursive Algorithm": 1.0,
        "Algorithmic Paradigm": 0.8,  # Parent
        "Backtracking": 0.7
    },
    "Graph Coloring": {
        "NP-Complete Problem": 0.7,
        "Graph Theory Problem": 0.8,
        "Constraint Satisfaction Problem": 0.5  # Can be framed as a CSP
    },

    # --- Categories (still needed for AI matching) ---
    "Search Algorithm": {
        "DFS": 0.7,
        "BFS": 0.7,
        "Backtracking": 0.7,
        "Heuristic Search": 0.7,
        "A* Search": 0.7
    },
    "Graph Theory Problem": {
        "Hamiltonian Path Problem": 0.8,
        "Graph Coloring": 0.8,
        "DFS": 0.6,
        "BFS": 0.6
    },
    "Computational Complexity Class": {
        "NP-Complete Problem": 0.8
    },
    "Algorithmic Paradigm": {
        "Divide and Conquer": 0.8,
        "Recursion": 0.8,
        "Constraint Satisfaction Problem": 0.5
    },
    "Discrete Mathematics": {
        "Hamiltonian Path Problem": 0.5
    },
    "Theoretical Computer Science": {
        "NP-Complete Problem": 0.5
    },
    "Heuristic Algorithm": {
        "Warnsdorff's Rule": 0.8,
        "Greedy Algorithm": 0.7,
        "Heuristic Search": 0.9
    },
    "Approximation Algorithm": {
        "Greedy Algorithm": 0.8
    },
    "Optimization Technique": {
        "Greedy Algorithm": 0.8
    },
    "Search Problem": {
        "Constraint Satisfaction Problem": 0.8,
        "NP-Complete Problem": 0.6
    },
    "A* Search": {
        "Heuristic Search": 0.9,
        "BFS": 0.6,
        "Search Algorithm": 0.7
    },
    "Iterative Algorithm": {
        "Recursive Algorithm": 0.3,  # Opposite
        "BFS": 0.8
    },

    # --- NEW: Complexity Concepts ---
    "O(N!)": {
        "Varies (Heuristic/Exponential)": 0.7,  # N! is a form of exponential
        "Exponential Complexity": 1.0,  # Synonym
        "NP-Complete": 0.5  # Many NP-Complete problems have factorial brute-force
    },
    "O(2^N)": {
        "Varies (Heuristic/Exponential)": 0.7,  # 2^N is a form of exponential
        "Exponential Complexity": 1.0,  # Synonym
        "NP-Complete": 0.5  # Many NP-Complete problems have 2^N complexity
    },
    "Varies (Heuristic/Exponential)": {
        "O(N!)": 0.7,
        "O(2^N)": 0.7,
        "Exponential Complexity": 0.8,
        "Heuristic Algorithm": 0.6  # Related concept
    },
    "NP-Complete": {
        # This one is already here, but let's link it to complexity
        "O(N!)": 0.5,
        "O(2^N)": 0.5,
        "Exponential Complexity": 0.6
    },
    "Exponential Complexity": {  # A new category
        "O(N!)": 1.0,
        "O(2^N)": 1.0,
        "Varies (Heuristic/Exponential)": 0.8,
        "NP-Complete": 0.6
    }
}


# --- 3. LOCAL QUESTION GENERATION ---
QUESTION_TEMPLATES: Dict[str, List[str]] = {
    "Type": [
        "In what category of problems does {problem} belong?",
        "How would you classify {problem} conceptually?",
        "Describe the nature of {problem} from a CS theory viewpoint."
    ],
    "Solutions": [
        "Name a technique that can solve {problem}.",
        "Which algorithmic strategy would you apply to tackle {problem}?",
        "Give one go-to method for addressing {problem}."
    ],
    "Complexity": [
        "What complexity do we typically associate with {problem}?",
        "Roughly how expensive is it to solve {problem}?",
        "State the asymptotic runtime usually cited for {problem}."
    ]
}


def generate_question_with_llama(problem_name: str, property_key: str, concept: str) -> Optional[str]:
    """
    Use the local llama.cpp model to craft a creative question. Falls back to
    None if the model is unavailable.
    """
    llama = get_llama()
    if llama is None:
        return None

    property_guidance = {
        "Type": "Ask explicitly for the classification/category/type of the problem.",
        "Solutions": "Ask for an algorithmic technique or strategy that solves the problem.",
        "Complexity": "Ask for the time complexity / asymptotic runtime and mention the word 'complexity'."
    }.get(property_key, "Ask for information matching the property key.")

    prompt = (
        "You are a witty CS professor writing exam questions.\n"
        f"Problem: {problem_name}\n"
        f"Target property: {property_key}\n"
        f"Expected concept: {concept}\n"
        f"Guidance: {property_guidance}\n"
        "Write exactly one self-contained question whose correct answer is the expected concept. "
        "The question must explicitly ask the student to NAME or DESCRIBE the concept (never yes/no). "
        "Keep it under 2 sentences.\n"
        "Question:\n"
    )
    try:
        response = llama(  # type: ignore[misc]
            prompt,
            max_tokens=128,
            temperature=0.75,
            stop=["\n\n", "Question:", "Answer:"]
        )
        text = response["choices"][0]["text"].strip()
        return text or None
    except Exception as exc:  # pragma: no cover
        print(f"[LLM Warning] Local model generation failed: {exc}")
        return None


def generate_question_text(problem_name: str, property_key: str, concept: str) -> str:
    """
    Generate a readable question either via the local LLM or, if unavailable,
    via deterministic templates.
    """
    llm_question = generate_question_with_llama(problem_name, property_key, concept)
    if llm_question:
        return llm_question

    templates = QUESTION_TEMPLATES.get(property_key)
    if templates:
        return random.choice(templates).format(problem=problem_name, concept=concept)

    return f"What is the {property_key.lower()} of {problem_name}, ideally touching on {concept}?"


# --- 4. AI SIMULATION (ANSWER PARSING) ---
def simulate_ai_semantic_match(user_answer: str, valid_concepts: List[str]) -> Optional[str]:
    """
    Simulates calling an LLM to perform semantic matching.
    This is the *only* concept extraction method.
    """
    system_prompt = "You are an expert in computer science terminology. Your job is to match the user's answer to the *best* canonical concept from the provided list. Respond with *only* the matching concept string. If no concept is a good match, respond with 'None'."

    user_prompt = f"""
    User Answer: "{user_answer}"
    Valid Canonical Concepts: {valid_concepts}
    Task: Which concept from the list is the *best* match for the user's answer?
    Best Match:
    """

    print(f"\n[AI Simulation] Calling LLM for SEMANTIC MATCH...")
    # time.sleep(0.5) # Simulate network delay

    # --- SIMULATION LOGIC ---
    # In a real app, the LLM would return its best match.
    # We'll simulate its "semantic understanding" with a few rules
    # and direct keyword checks for concepts.

    user_answer_lower = user_answer.lower()

    # 1. Check for canonical concepts directly (handles synonyms)
    # This simulation is "smart" and checks our main concepts
    # It replaces the need for the old `extract_concept`

    # Sort by length to find "Hamiltonian Path Problem" before "Search Problem"
    sorted_concepts = sorted(valid_concepts, key=len, reverse=True)

    # --- FIX ---
    # Pass 1: Check for exact (or near-exact) canonical concept matches first.
    # This prioritizes "Search Problem" over "Search Algorithm" if the user
    # types "search problem".
    for concept in sorted_concepts:
        if concept.lower() in user_answer_lower:
            return concept

    # Pass 2: If no exact match, check for partial/cleaned concepts
    for concept in sorted_concepts:
        # Check for synonyms or partials
        # This is a "cheat" for the simulation, but represents what the AI would do
        concept_clean = concept.lower().split(" ")[0].split("(")[0]  # e.g., "hamiltonian", "o(n!)" -> "o"
        if concept.lower() in user_answer_lower:
            return concept  # This line is now redundant from Pass 1, but harmless

        # --- FIX ---
        # Changed from > 3 to > 1 to allow "A*", "DFS", "BFS"
        # but still block the single "o" from "O(N!)"
        if concept_clean in user_answer_lower and len(concept_clean) > 1:
            return concept

    # 2. Check for semantic phrases
    if "deep" in user_answer_lower and "DFS" in valid_concepts:
        return "DFS"
    if "broad" in user_answer_lower and "BFS" in valid_concepts:
        return "BFS"
    if "2 power n" in user_answer_lower and "O(2^N)" in valid_concepts:
        return "O(2^N)"
    if "n factorial" in user_answer_lower and "O(N!)" in valid_concepts:
        return "O(N!)"
    if "heuristic" in user_answer_lower and "Heuristic Search" in valid_concepts:
        return "Heuristic Search"

    # Fallback: if no rule matches, simulate no match
    print("[AI Simulation] No high-confidence semantic match found.")
    return None
    # ------------------------


# --- 5. QUESTION GENERATION ---
def generate_question(problem_name: str, property_key: str) -> Dict[str, Any]:
    """
    Generates a question using the local LLM (with template fallback).
    """
    problem_data = KNOWLEDGE_GRAPH[problem_name]
    expected_concepts = problem_data[property_key]  # Could be list or str

    if isinstance(expected_concepts, list):
        if not expected_concepts:
            raise ValueError(f"No expected concepts configured for {problem_name} -> {property_key}.")
        sim_prompt_concept = random.choice(expected_concepts)
    else:
        sim_prompt_concept = expected_concepts

    question_text = generate_question_text(problem_name, property_key, sim_prompt_concept)

    return {
        "problem": problem_name,
        "property_key": property_key,
        "expected_concepts": expected_concepts,
        "question": question_text
    }


# --- 6. ACCURATE GRADING LOGIC ---
def grade_answer(question_data: Dict[str, str], user_answer: str) -> Dict[str, Any]:
    """
    Grades the user's answer using the AI semantic match and weighted graph.
    """
    expected_concepts = question_data["expected_concepts"]  # This is now a list or a string
    property_key = question_data["property_key"]

    # --- AI-Only Concept Extraction ---

    # 1. Build the list of all valid concepts for the AI to check against
    valid_concepts = list(SEMANTIC_LINKS.keys())
    # Add complexity strings
    for k in KNOWLEDGE_GRAPH:
        valid_concepts.append(KNOWLEDGE_GRAPH[k]["Complexity"])
    valid_concepts = list(set(valid_concepts))  # De-duplicate

    # 2. Call the AI simulation
    extracted_user_concept = simulate_ai_semantic_match(user_answer, valid_concepts)

    feedback = ""
    score = 0.0

    # --- GRADING ---
    if extracted_user_concept is None:
        feedback = "The answer did not contain a recognizable concept from our knowledge base."
        return {"score": 0.0, "feedback": feedback, "expected": str(expected_concepts), "user_extracted": None}

    # Create a list regardless of what was passed
    if not isinstance(expected_concepts, list):
        expected_concepts_list = [expected_concepts]  # Make it a list of one
    else:
        expected_concepts_list = expected_concepts

    # --- Scenario A: Exact Match ---
    # Check if the extracted concept is an exact match for *any* of the expected concepts

    # We need to normalize for case-insensitive comparison
    normalized_expected = [str(c).lower() for c in expected_concepts_list]

    if extracted_user_concept.lower() in normalized_expected:
        score = 1.0
        # Find the original cased concept for feedback
        correct_concept = expected_concepts_list[normalized_expected.index(extracted_user_concept.lower())]
        feedback = f"Correct! '{correct_concept}' is a valid solution."

    # --- Scenario B: Partial Credit (Weighted Graph) ---
    # --- MODIFICATION ---
    # We now check *all* question types for partial credit.
    # The 'elif' condition limiting this to "Solutions" or "Type" is removed.
    else:
        best_score = 0.0
        best_match_concept = ""

        # Check for semantic links against *all* expected concepts
        for expected_concept in expected_concepts_list:
            current_score = 0.0

            # Check forward link
            if expected_concept in SEMANTIC_LINKS and extracted_user_concept in SEMANTIC_LINKS[expected_concept]:
                current_score = SEMANTIC_LINKS[expected_concept][extracted_user_concept]

            # Check reverse link
            elif extracted_user_concept in SEMANTIC_LINKS and expected_concept in SEMANTIC_LINKS[
                extracted_user_concept]:
                current_score = SEMANTIC_LINKS[extracted_user_concept][expected_concept]

            if current_score > best_score:
                best_score = current_score
                best_match_concept = expected_concept

        if best_score > 0.0:
            score = best_score
            feedback = (
                f"Partial Credit ({score * 100}%). Your answer, '{extracted_user_concept}', is conceptually related to '{best_match_concept}'. "
                f"Full credit answers include: {', '.join(expected_concepts_list)}."
            )
        else:
            score = 0.0
            feedback = (
                f"Incorrect. While '{extracted_user_concept}' is a valid concept, it is not "
                f"semantically related to the expected answer(s) ({', '.join(expected_concepts_list)})."
            )

    # --- Scenario C: No Partial Credit (e.g., "Complexity") ---
    # --- MODIFICATION ---
    # This block is now redundant because Scenario B handles all non-exact matches.
    # else:
    #     ... (This block has been removed) ...

    return {
        "score": score,
        "feedback": feedback,
        "expected": ', '.join(expected_concepts_list),  # Show all expected for feedback
        "user_extracted": extracted_user_concept
    }


# --- 7. INTERACTIVE TERMINAL LOOP ---
def main():
    """
    Runs an INTERACTIVE loop to allow the user to answer
    randomly generated questions.
    """
    print("--- Knowledge Graph-Based Grader (Static Graph) ---")
    print("Type 'exit' or 'quit' to stop.")

    # --- NEW: Create and shuffle all possible questions ---
    available_questions = []
    for problem_name in KNOWLEDGE_GRAPH.keys():
        for prop_key in ["Type", "Solutions", "Complexity"]:
            available_questions.append((problem_name, prop_key))

    random.shuffle(available_questions)
    max_questions = len(available_questions)

    # --- NEW: Get number of questions from user ---
    num_questions = 0
    while True:
        try:
            user_input = input(f"How many questions would you like? (1-{max_questions}): ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting.")
                return  # Exit the main function

            num_questions = int(user_input)
            if 0 < num_questions <= max_questions:
                break  # Valid input
            else:
                print(f"Please enter a number between 1 and {max_questions}.")
        except ValueError:
            print("That's not a valid number. Please try again.")

    # --- MODIFIED: Loop for the requested number of questions ---
    for q_index in range(num_questions):
        try:
            print("\n" + "=" * 50)
            print(f"Question {q_index + 1} of {num_questions}")  # Add counter

            # 1. Get the non-repeating question
            problem_name, property_key = available_questions[q_index]

            # 2. Generate the question text
            q_data = generate_question(problem_name, property_key)

            print(f"Problem: {q_data['problem']} | Property: {q_data['property_key']}")
            print("-" * 50)
            print(f"QUESTION: {q_data['question']}")

            # 2. Get user input
            user_input = input("Your Answer: ")

            # 3. Check for exit condition
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting interactive session.")
                break

            # 4. Grade the answer
            result = grade_answer(q_data, user_input)

            # 5. Provide feedback
            print("-" * 50)
            print(f"SCORE: {result['score'] * 100}%")
            print(f"FEEDBACK: {result['feedback']}")
            print(f"(Extracted: '{result.get('user_extracted')}' | Expected: '{result.get('expected')}')")

        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("Skipping to next question...")
            time.sleep(1)  # Brief pause before continuing

    print("\n" + "=" * 50)
    print("Quiz complete! Thanks for playing.")
    print("=" * 50)


# This standard Python construct ensures that main() runs
# only when the script is executed directly (not imported).
if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
