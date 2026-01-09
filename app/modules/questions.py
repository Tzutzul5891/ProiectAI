import random


class QuestionGenerator:
    def __init__(self):
        self.question = ""
        self.correct_answer = ""
        self.question_type = ""
        self.solution_type = ""
        self.subject = ""

        # Predefined questions for different subjects
        self.ai_questions = [
            {
                "question": "Ce este învățarea supervizată în contextul inteligenței artificiale?",
                "answer": "Învățarea supervizată este un tip de machine learning unde modelul este antrenat pe date etichetate, adică date pentru care știm deja răspunsul corect. Modelul învață să facă predicții bazate pe exemple de intrare-ieșire."
            },
            {
                "question": "Explică diferența între deep learning și machine learning tradițional.",
                "answer": "Deep learning folosește rețele neuronale cu multe straturi (deep) pentru a învăța reprezentări complexe din date. Machine learning tradițional folosește algoritmi mai simpli și necesită mai multă preprocesare manuală a datelor."
            },
            {
                "question": "Ce este un algoritm de căutare în spațiul stărilor?",
                "answer": "Un algoritm de căutare în spațiul stărilor explorează diferite stări posibile ale unei probleme pentru a găsi o soluție. Exemple includ BFS, DFS, A* care explorează graful stărilor sistematic."
            },
            {
                "question": "Descrie conceptul de overfitting în machine learning.",
                "answer": "Overfitting apare când un model învață prea bine datele de antrenare, inclusiv zgomotul, și nu generalizează bine pe date noi. Modelul are performanță excelentă pe datele de antrenare dar slabă pe date de test."
            }
        ]

        self.complexity_questions = [
            {
                "question": "Ce înseamnă complexitatea temporală O(n log n)?",
                "answer": "Complexitatea O(n log n) înseamnă că timpul de execuție crește proporțional cu n log n, unde n este dimensiunea inputului. Algoritmi precum merge sort și heap sort au această complexitate."
            },
            {
                "question": "Explică diferența între complexitatea temporală și spațială.",
                "answer": "Complexitatea temporală măsoară timpul de execuție al unui algoritm în funcție de dimensiunea inputului. Complexitatea spațială măsoară memoria folosită de algoritm. Un algoritm poate fi rapid dar să folosească multă memorie sau invers."
            },
            {
                "question": "Ce este complexitatea în cel mai rău caz (worst case)?",
                "answer": "Complexitatea în cel mai rău caz analizează performanța algoritmului pentru inputul care necesită cel mai mult timp sau memorie. Este importantă pentru a garanta că algoritmul va funcționa bine în orice situație."
            },
            {
                "question": "Descrie conceptul de NP-completitudine.",
                "answer": "O problemă este NP-completă dacă este în NP (verificabilă în timp polinomial) și orice problemă din NP poate fi redusă la ea în timp polinomial. Problemele NP-complete sunt considerate dificile de rezolvat eficient."
            }
        ]

        self.ai_multiple_choice = [
            {
                "question": "Care dintre următoarele este un algoritm de căutare uninformat?",
                "options": ["A*", "BFS", "Greedy Best-First", "Hill Climbing"],
                "correct": ["BFS", "Hill Climbing"],
                "answer": "BFS și Hill Climbing sunt algoritmi uninformați. A* și Greedy Best-First sunt informați (folosesc euristică)."
            },
            {
                "question": "Ce tipuri de învățare există în machine learning?",
                "options": ["Supervizată", "Nesupervizată", "Prin întărire", "Toate cele de mai sus"],
                "correct": ["Toate cele de mai sus"],
                "answer": "Machine learning include învățare supervizată (date etichetate), nesupervizată (fără etichete) și prin întărire (reward-based)."
            }
        ]

        self.complexity_multiple_choice = [
            {
                "question": "Care dintre următoarele algoritmi au complexitate O(n²)?",
                "options": ["Bubble Sort", "Merge Sort", "Insertion Sort", "Quick Sort (worst case)"],
                "correct": ["Bubble Sort", "Insertion Sort", "Quick Sort (worst case)"],
                "answer": "Bubble Sort, Insertion Sort și Quick Sort în cel mai rău caz au complexitate O(n²). Merge Sort are O(n log n)."
            },
            {
                "question": "Ce complexitate are căutarea binară?",
                "options": ["O(n)", "O(log n)", "O(n log n)", "O(1)"],
                "correct": ["O(log n)"],
                "answer": "Căutarea binară are complexitate O(log n) deoarece la fiecare pas elimină jumătate din spațiul de căutare."
            }
        ]

    def generate_problem(self, subject, solution_type, custom_question=None, custom_answer=None):
        """
        Generate a question based on subject and solution type.

        Args:
            subject: "Inteligenta Artificiala", "Complexitate", or "Subiect Personalizat"
            solution_type: "Intrebare cu raspuns multiplu" or "Definitie de scris"
            custom_question: Required if subject is "Subiect Personalizat"
            custom_answer: Required if subject is "Subiect Personalizat"
        """
        self.subject = subject
        self.solution_type = solution_type

        if subject == "Subiect Personalizat":
            if not custom_question or not custom_answer:
                return None, "Eroare: Te rog completează întrebarea și răspunsul corect pentru subiectul personalizat."
            self.question = custom_question
            self.correct_answer = custom_answer
            return {"question": self.question, "type": solution_type}, self.correct_answer

        # Generate question for predefined subjects
        if solution_type == "Definitie de scris":
            if subject == "Inteligenta Artificiala":
                selected = random.choice(self.ai_questions)
            elif subject == "Complexitate":
                selected = random.choice(self.complexity_questions)
            else:
                return None, "Subiect necunoscut."

            self.question = selected["question"]
            self.correct_answer = selected["answer"]
            return {"question": self.question, "type": solution_type}, self.correct_answer

        elif solution_type == "Intrebare cu raspuns multiplu":
            if subject == "Inteligenta Artificiala":
                selected = random.choice(self.ai_multiple_choice)
            elif subject == "Complexitate":
                selected = random.choice(self.complexity_multiple_choice)
            else:
                return None, "Subiect necunoscut."

            self.question = selected["question"]
            self.correct_answer = selected["answer"]
            return {
                "question": self.question,
                "options": selected["options"],
                "correct": selected["correct"],
                "type": solution_type
            }, self.correct_answer

        return None, "Tip de soluție necunoscut."
