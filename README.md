# SmarTest - Proiect IA (Streamlit)

SmarTest este o aplicație locală pentru generare de probleme tip examen și evaluare automată. Constrângere de design: **fără apeluri la API-uri LLM în runtime**; evaluarea și soluțiile sunt **locale/deterministe**.

## ✅ Implementat acum

- **Generare probleme (local):**
  - **Jocuri:** matrice 2x2 + detectare Echilibru Nash pur (`app/modules/games.py`).
  - **Căutare:** N-Queens, Turul Calului (5x5/6x6), Turnurile din Hanoi (3/4 tije, 3–5 discuri) (`app/modules/search.py`).
- **UI Streamlit (interactiv):**
  - tablă interactivă N-Queens, Turul Calului și Turnurile din Hanoi (`app/gui/components.py`).
- **Mod Test (multi-întrebări):**
  - selectezi capitole/subiecte + N întrebări, navighezi între ele și poți exporta PDF separat pentru test vs answer key (`main.py`, `app/utils/pdf_generator.py`).
- **Evaluare (local/determinist):**
  - Nash: scor semantic pe explicație + verificare coordonate (regex) (`app/evaluator/semantic.py`).
  - N-Queens: validare exactă a configurației + scor parțial.
  - Turul Calului: validare mișcări + scor euristic local.
  - Hanoi: validare corectitudine + eficiență față de optim.
- **Export PDF:** generare subiect PDF (`app/utils/pdf_generator.py`).

## 🔌 Convenție: ce returnează un generator

Convenția nouă (recomandată) este ca fiecare generator să întoarcă un `ProblemInstance` (vezi `app/modules/base_problem.py`):

- `data`: conținut structurat pentru UI/PDF (tablă, matrice etc.)
- `prompt`: enunțul problemei
- `solution`: soluția în format structurat (dacă există)
- `explanation`: explicația gold standard (text)
- `metadata`: câmpuri extra (dimensiuni, start_pos, număr mutări optime etc.)

Pentru compatibilitate cu UI-ul curent, clasele expun în continuare `generate_problem()` (legacy) care întoarce `(data, explanation)`.

## 🧱 Modele: `Question` & `TestSession`

- `Question` împachetează enunțul + datele + answer key (răspuns/expl.) într-un singur obiect: `app/models/test_session.py`.
- `TestSession` ține o listă de `Question` + index curent + răspunsuri/scoruri (pentru teste cu N întrebări).

Exemplu `Question` serializat (dict): `app/models/test_session.py` (`EXAMPLE_QUESTION_DICT`). Pentru PDF, poți folosi `Question.pdf_kwargs()` și `create_pdf(**kwargs)`.

```py
from app.models.test_session import EXAMPLE_QUESTION_DICT
from app.utils.pdf_generator import create_pdf

pdf_bytes = create_pdf(
    problem_type=EXAMPLE_QUESTION_DICT["metadata"]["ui_label"],
    requirement=EXAMPLE_QUESTION_DICT["prompt_text"],
    matrix_data=EXAMPLE_QUESTION_DICT["data"],
)
```

## 🛠️ Rulare

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

## ⚙️ Opțiuni offline / determinism

- **Fără download de modele (offline strict):** `SMARTEST_LOCAL_MODELS_ONLY=1`
- **Model SBERT local (path):** `SMARTEST_SBERT_MODEL=/cale/către/model`
- **Generare reproductibilă (seed):** `SMARTEST_SEED=42`

Evaluatorul semantic încearcă SBERT local; dacă nu poate încărca modelul, folosește un fallback lexical determinist.

## 🗂️ Structură proiect

```
ProiectAI/
├── app/
│   ├── evaluator/      # evaluare (semantic, exact)
│   ├── gui/            # componente UI Streamlit
│   ├── modules/        # generatoare de probleme (search, games, viitor: csp/adversarial)
│   └── utils/          # utilitare (PDF, helpers)
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

## 🧭 Ce urmează

- Mutarea logicii de enunț/PDF pe `ProblemInstance.prompt` (mai puțin duplicat în `main.py`).
- Implementări reale în `app/modules/csp.py` și `app/modules/adversarial.py`.
- `app/utils/pdf_parser.py`: parsare PDF -> structură internă (dacă e necesar).
- Teste minimale pentru generatoare/evaluatori (local, determinist).
